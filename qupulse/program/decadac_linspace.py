import abc
import contextlib
import dataclasses
import numpy as np
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, ContextManager, Iterable, Tuple, Union, Dict, List, Iterator, Callable

from qupulse import ChannelID
from qupulse._program import Waveform, TimeType

# this resolution is used to unify increments
# the increments themselves remain floats
DEFAULT_INCREMENT_RESOLUTION: float = 1e-9


@dataclass(frozen=True)
class DepKey:
    """The key that identifies how a certain set command depends on iteration indices. The factors are rounded with a
    given resolution to be independent on rounding errors.

    These objects allow backends which support it to track multiple amplitudes at once.
    """
    factors: Tuple[int, ...]

    @classmethod
    def from_voltages(cls, voltages: Sequence[float], resolution: float):
        # remove trailing zeros
        while voltages and voltages[-1] == 0:
            voltages = voltages[:-1]
        return cls(tuple(int(round(voltage / resolution)) for voltage in voltages))


@dataclass
class LinSpaceNode:
    """AST node for a program that supports linear spacing of set points as well as nested sequencing and repetitions"""

    def dependencies(self) -> Mapping[int, set]:
        raise NotImplementedError


@dataclass
class LinSpaceHold(LinSpaceNode):
    """Hold voltages for a given time. The voltages and the time may depend on the iteration index."""

    bases: Tuple[float, ...]
    factors: Tuple[Optional[Tuple[float, ...]], ...]

    duration_base: TimeType
    duration_factors: Optional[Tuple[TimeType, ...]]

    def dependencies(self) -> Mapping[int, set]:
        return {idx: {factors}
                for idx, factors in enumerate(self.factors)
                if factors}


@dataclass
class LinSpaceArbitraryWaveform(LinSpaceNode):
    """This is just a wrapper to pipe arbitrary waveforms through the system."""
    waveform: Waveform
    channels: Tuple[ChannelID, ...]


@dataclass
class LinSpaceRepeat(LinSpaceNode):
    """Repeat the body count times."""
    body: Tuple[LinSpaceNode, ...]
    count: int

    def dependencies(self):
        dependencies = {}
        for node in self.body:
            for idx, deps in node.dependencies().items():
                dependencies.setdefault(idx, set()).update(deps)
        return dependencies


@dataclass
class LinSpaceIter(LinSpaceNode):
    """Iteration in linear space are restricted to range 0 to length.

    Offsets and spacing are stored in the hold node."""
    body: Tuple[LinSpaceNode, ...]
    length: int

    def dependencies(self):
        dependencies = {}
        for node in self.body:
            for idx, deps in node.dependencies().items():
                # remove the last elemt in index because this iteration sets it -> no external dependency
                shortened = {dep[:-1] for dep in deps}
                if shortened != {()}:
                    dependencies.setdefault(idx, set()).update(shortened)
        return dependencies


@dataclass
class LoopLabel:
    idx: int
    count: int


@dataclass
class Increment:
    channel: int
    value: float
    dependency_key: DepKey


@dataclass
class Set:
    channel: int
    value: float
    key: DepKey = dataclasses.field(default_factory=lambda: DepKey(()))


@dataclass
class Wait:
    duration: TimeType


@dataclass
class LoopJmp:
    idx: int


@dataclass
class Play:
    waveform: Waveform
    channels: Tuple[ChannelID]


Command = Union[Increment, Set, LoopLabel, LoopJmp, Wait, Play]


@dataclass(frozen=True)
class DepState:
    base: float
    iterations: Tuple[int, ...]

    def required_increment_from(self, previous: 'DepState', factors: Sequence[float]) -> float:
        assert len(self.iterations) == len(previous.iterations)
        assert len(self.iterations) == len(factors)

        increment = self.base - previous.base
        for old, new, factor in zip(previous.iterations, self.iterations, factors):
            # By convention there are only two possible values for each integer here: 0 or the last index
            # The three possible increments are none, regular and jump to next line

            if old == new:
                # we are still in the same iteration of this sweep
                pass

            elif old < new:
                assert old == 0
                # regular iteration, although the new value will probably be > 1, the resulting increment will be
                # applied multiple times so only one factor is needed.
                increment += factor

            else:
                assert new == 0
                # we need to jump back. The old value gives us the number of increments to reverse
                increment -= factor * old
        return increment


@dataclass
class _TranslationState:
    """This is the state of a translation of a LinSpace program to a command sequence."""

    label_num: int = dataclasses.field(default=0)
    commands: List[Command] = dataclasses.field(default_factory=list)
    iterations: List[int] = dataclasses.field(default_factory=list)
    active_dep: Dict[int, DepKey] = dataclasses.field(default_factory=dict)
    dep_states: Dict[int, Dict[DepKey, DepState]] = dataclasses.field(default_factory=dict)
    plain_voltage: Dict[int, float] = dataclasses.field(default_factory=dict)
    resolution: float = dataclasses.field(default_factory=lambda: DEFAULT_INCREMENT_RESOLUTION)

    def new_loop(self, count: int):
        label = LoopLabel(self.label_num, count)
        jmp = LoopJmp(self.label_num)
        self.label_num += 1
        return label, jmp

    def get_dependency_state(self, dependencies: Mapping[int, set]):
        return {
            self.dep_states.get(ch, {}).get(DepKey.from_voltages(dep, self.resolution), None)
            for ch, deps in dependencies.items()
            for dep in deps
        }

    def set_voltage(self, channel: int, value: float):
        key = DepKey(())
        if self.active_dep.get(channel, None) != key or self.plain_voltage.get(channel, None) != value:
            self.commands.append(Set(channel, value, key))
            self.active_dep[channel] = key
            self.plain_voltage[channel] = value

    def _add_repetition_node(self, node: LinSpaceRepeat):
        pre_dep_state = self.get_dependency_state(node.dependencies())
        label, jmp = self.new_loop(node.count)
        initial_position = len(self.commands)
        self.commands.append(label)
        self.add_node(node.body)
        post_dep_state = self.get_dependency_state(node.dependencies())
        if pre_dep_state != post_dep_state:
            # hackedy
            self.commands.pop(initial_position)
            self.commands.append(label)
            label.count -= 1
            self.add_node(node.body)
        self.commands.append(jmp)

    def _add_iteration_node(self, node: LinSpaceIter):
        self.iterations.append(0)
        self.add_node(node.body)

        if node.length > 1:
            self.iterations[-1] = node.length
            label, jmp = self.new_loop(node.length - 1)
            self.commands.append(label)
            self.add_node(node.body)
            self.commands.append(jmp)
        self.iterations.pop()

    def _set_indexed_voltage(self, channel: int, base: float, factors: Sequence[float]):
        dep_key = DepKey.from_voltages(voltages=factors, resolution=self.resolution)
        new_dep_state = DepState(
            base,
            iterations=tuple(self.iterations)
        )

        current_dep_state = self.dep_states.setdefault(channel, {}).get(dep_key, None)
        if current_dep_state is None:
            assert all(it == 0 for it in self.iterations)
            self.commands.append(Set(channel, base, dep_key))
            self.active_dep[channel] = dep_key

        else:
            inc = new_dep_state.required_increment_from(previous=current_dep_state, factors=factors)

            # we insert all inc here (also inc == 0) because it signals to activate this amplitude register
            if inc or self.active_dep.get(channel, None) != dep_key:
                self.commands.append(Increment(channel, inc, dep_key))
            self.active_dep[channel] = dep_key
        self.dep_states[channel][dep_key] = new_dep_state

    def _add_hold_node(self, node: LinSpaceHold):
        if node.duration_factors:
            raise NotImplementedError("TODO")

        for ch, (base, factors) in enumerate(zip(node.bases, node.factors)):
            if factors is None:
                self.set_voltage(ch, base)
                continue

            else:
                self._set_indexed_voltage(ch, base, factors)

        self.commands.append(Wait(node.duration_base))

    def add_node(self, node: Union[LinSpaceNode, Sequence[LinSpaceNode]]):
        """Translate a (sequence of) linspace node(s) to commands and add it to the internal command list."""
        if isinstance(node, Sequence):
            for lin_node in node:
                self.add_node(lin_node)

        elif isinstance(node, LinSpaceRepeat):
            self._add_repetition_node(node)

        elif isinstance(node, LinSpaceIter):
            self._add_iteration_node(node)

        elif isinstance(node, LinSpaceHold):
            self._add_hold_node(node)

        elif isinstance(node, LinSpaceArbitraryWaveform):
            self.commands.append(Play(node.waveform, node.channels))

        else:
            raise TypeError("The node type is not handled", type(node), node)


def to_increment_commands(linspace_nodes: Sequence[LinSpaceNode]) -> List[Command]:
    """translate the given linspace node tree to a minimal sequence of set and increment commands as well as loops."""
    state = _TranslationState()
    state.add_node(linspace_nodes)
    return state.commands


def reduce_consecutive_waits(commands:List[Command]) -> Tuple[bool, List[Command]]:

    changed_something = False
    new_commands = []
    i = 0
    while i < len(commands):

        if commands[i].__class__.__name__ == "Wait":
            consec_waits = [commands[i]]
            # select all consecutive waits
            j = i+1
            while j < len(commands) and commands[j].__class__.__name__ == "Wait":
                consec_waits.append(commands[j])
                j += 1

            # if we have only one wait. append that wait to the new command list
            if len(consec_waits) == 1:
                new_commands.append(commands[i])
            else:
                total_wait_time = np.sum([c.duration for c in consec_waits])
                new_commands.append(Wait(total_wait_time))
                i = j-1 # i should now be the index of the last Wait that we looked at
                assert commands[i] == consec_waits[-1]
                changed_something = True
        else:
            new_commands.append(commands[i])

        i += 1

    return changed_something, new_commands

def reduce_looped_waits(commands:List[Command]) -> Tuple[bool, List[Command]]:

    changed_something = False
    new_commands = []
    i = 0
    while i < len(commands):

        if commands[i].__class__.__name__ == "LoopLabel":
            nested_commands = []
            # select all consecutive waits
            j = i+1
            while j < len(commands) and commands[j].__class__.__name__ == "Wait" and ((not (commands[j].__class__.__name__ == "LoopJmp") or (int(commands[j].idx) != int(commands[i].idx)))):
                nested_commands.append(commands[j])
                j += 1

            # check if the loop contains only one wait
            if commands[j].__class__.__name__ == "LoopJmp" and len(nested_commands) == 1 and nested_commands[0].__class__.__name__ == "Wait":

                # then create a new wait command
                total_wait_time = nested_commands[0].duration * commands[i].count
                new_commands.append(Wait(total_wait_time))
                i = j # i should now point to the loop jump. The following assert does not cover it completely, but better than nothing.
                assert commands[i].__class__.__name__ == "LoopJmp"
                changed_something = True
            else:
                new_commands.append(commands[i])
        else:
            new_commands.append(commands[i])

        i += 1

    return changed_something, new_commands


def reduce_zero_increments(commands:List[Command]) -> Tuple[bool, List[Command]]:

    changed_something = False
    new_commands = []
    i = 0
    while i < len(commands):

        if commands[i].__class__.__name__ == "Increment":
            if commands[i].value != 0:
                new_commands.append(commands[i])
            else: 
                changed_something = True
        else:
            new_commands.append(commands[i])

        i += 1

    return changed_something, new_commands




def reduce_commands(commands:List[Command], functions_for_reduction:Union[List[Callable], None]=None) -> List[Command]:
    """ This function reduces commands. E.g. waits following waits will be turned into one wait. Loops over just waits, will be combined into one wait.
    """

    if functions_for_reduction is None:
        functions_for_reduction = [
            reduce_zero_increments,
            reduce_consecutive_waits, 
            reduce_looped_waits, 
        ]

    max_tries = 100_000_000
    n_tries = 0
    some_change = True
    while some_change and n_tries <= max_tries:
        # go through all the functions and reduce the program
        change = False
        for fn in functions_for_reduction:
            c, commands = fn(commands)
            change = change or c
        n_tries += 1

        some_change = change

    return commands


