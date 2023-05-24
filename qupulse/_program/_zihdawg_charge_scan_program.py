# -*- coding: utf-8 -*-

import textwrap
from abc import ABC, abstractmethod
from typing import Tuple, Iterator, List, Iterable, Dict
from zhinst.toolkit import CommandTable
from qupulse._program._loop import make_compatible
from qupulse.pulses.pulse_template import PulseTemplate
from qupulse.pulses import ForLoopPT, PointPT, SequencePT, AtomicMultiChannelPT
from qupulse.hardware.awgs.zihdawg import  HDAWGChannelGroup
from qupulse._program.seqc import HDAWGProgramEntry, WaveformMemory, BinaryWaveform, ConcatenatedWaveform
from qupulse.hardware.awgs.base import AWGAmplitudeOffsetHandling
from qupulse.hardware.setup import PlaybackChannel, MarkerChannel, HardwareSetup
from qupulse.hardware.dacs.alazar import AlazarCard
from qupulse.hardware.setup import MeasurementMask
from atsaverage.operations import Downsample, ChunkedAverage

import time
import json

import numpy as np
import matplotlib.pyplot as plt
# import gridspec


#TODO: 50 or 1MOhm termination???

class FixedStructureProgram(ABC):
    
    CHANNELSTRING = 'ABCDEFGH'
    NEW_MIN_QUANT = 32
    FILE_NAME_TEMPLATE = '{hash}.csv'
    ALAZAR_CHANNELS = "ABCD"
    MEASUREMENT_NAME_TEMPLATE = 'M{i}'
    
    def __init__(self, 
                 name: str,
                 final_pt: PulseTemplate,
                 parameters: dict,
                 measurement_channels: List[str],
                 awg_object: HDAWGChannelGroup,
                 hardware_setup: HardwareSetup,
                 dac_object: AlazarCard, #or possibly other dac object?
                 auto_register: bool = True,
                 ):
        
        self._name = name
        self._awg = awg_object
        self._dac = dac_object
        self._hardware_setup = hardware_setup
        self._seqc_body = None
        
        self._loop_obj = final_pt.create_program(parameters=parameters)
        
        self.original = self._loop_obj
        
        self._measurement_channels = measurement_channels
        if len(self._measurement_channels) != 0 and self._dac is not None:
            self.register_measurements(measurement_channels, list(final_pt.measurement_names))
        self._measurement_result = None
        
        loop_method_list = [method_name for method_name in dir(self._loop_obj.__class__) if callable(getattr(self._loop_obj.__class__, method_name)) and not method_name.startswith("__")]
        loop_attribute_list = [method_name for method_name in dir(self._loop_obj.__class__) if not callable(getattr(self._loop_obj.__class__, method_name)) and not method_name.startswith("__")]

        for name in loop_method_list:
            self.add_loop_method(name)
        for name in loop_attribute_list:
            self.add_loop_attribute(name)
        

        #TODO: remove if already registered (should be just .remove_prgoram); but is it clean? - seems to work
        #TODO: run_callback - seems to work 
        #TODO: if not auto registered with this run_callback, plot func will not display correct result as not measure_program called automatically?
        #TODO: some weird error about "arming program without operations".... appears when adding with same name. perhaps some bug in qupulse/alazar.py?

        if auto_register:
            self._hardware_setup.remove_program(self.name)
            self._hardware_setup.register_program(self.name,self,run_callback=self.run_func)
        
        return
    
    def add_loop_method(self,method_name):
        setattr(self.__class__,method_name,eval('self._loop_obj.'+method_name))
        return
    
    def add_loop_attribute(self,attribute_name):
        setattr(self.__class__,attribute_name,eval('self._loop_obj.'+attribute_name))
        return
    
    def run_func(self):
        print("I'm executed")
        self._awg.run_current_program()
        if len(self._measurement_channels) != 0 and self._dac is not None:
            # self._measurement_result = self._dac.measure_program(self._measurement_channels)
            #!!! only works with measure_program copied from alazar2, where channels=None->self-inferred
            self._measurement_result = self._dac.measure_program()

    @property
    def name(self) -> str:
        return self._name
    
    @property
    def measurement_result(self) -> dict:
        return self._measurement_result
    
    @property
    @abstractmethod
    def corresponding_pt(self) -> Tuple[PulseTemplate,dict]:
        pass
    
    @property
    @abstractmethod
    def waveform_dict(self) -> dict:
        pass
    
    @abstractmethod
    def expand_ct(self,
            ct: CommandTable,
            starting_index: int
            ) -> CommandTable:
        pass
    
    @abstractmethod
    def wf_definitions_iter(self, indent: str) -> Iterator[Tuple[str,str]]:
        pass
    
    @abstractmethod
    def get_seqc_program_body(self, pos_var_name: str) -> str:
        pass
    
    @abstractmethod
    def plot_all_measurements(self):
        pass
    
    def measure_program(self) -> dict:
        self._measurement_result = self._dac.measure_program()
        return self.measurement_result
        
    #TODO: something with measurements
    #TODO: other op. than Downsample?
    def register_measurements(self,
                              measurement_channels: List[str], #from ["A","B","C","D"]
                              mask_list: Iterable[str],
                              ):
        
        operations = []
        self._dac.update_settings = True
        #!!! DONT USE i; CAUSE LIST SHUFFLED
        for i,mask in enumerate(mask_list):
            for channel in measurement_channels:
                id_string = f'{mask}{channel}'
                self._dac.register_mask_for_channel(id_string, self.ALAZAR_CHANNELS.find(channel))
                operations.append(Downsample(id_string,id_string))
                # operations.append(ChunkedAverage(id_string,id_string,chunkSize=100))

        for i,mask in enumerate(mask_list):
            self._hardware_setup.set_measurement(mask,
                                                 [MeasurementMask(self._dac, f'{mask}{channel}') for channel in measurement_channels],
                                                 allow_multiple_registration=True)
            
        self._dac.register_operations(self.name,operations)
        self._dac.update_settings = True #in MB's example set again at this stage; any difference?
            
    #TODO:
    def pt_to_binaries(self,pt,parameters,
                       ) -> tuple[BinaryWaveform]:

        def get_default_info(awg):
            return ([None] * awg.num_channels,
                    [None] * awg.num_channels,
                    [None] * awg.num_markers)
        
        playback_ids, voltage_trafos, marker_ids = get_default_info(self._awg)
        
        for channel_id in pt.defined_channels:
            for single_channel in self._hardware_setup._channel_map[channel_id]:
                if isinstance(single_channel, PlaybackChannel):
                    playback_ids[single_channel.channel_on_awg] = channel_id
                    voltage_trafos[single_channel.channel_on_awg] = single_channel.voltage_transformation
                elif isinstance(single_channel, MarkerChannel):
                    marker_ids[single_channel.channel_on_awg] = channel_id

        loop = pt.create_program(parameters=parameters)
        
        # from awg.upload
        # Go to qupulse nanoseconds time base.
        q_sample_rate = self._awg.sample_rate / 10**9

        # Adjust program to fit criteria.
        #TODO: is this required/favorable?
        make_compatible(loop,
                        minimal_waveform_length=self._awg.MIN_WAVEFORM_LEN,
                        waveform_quantum=self._awg.WAVEFORM_LEN_QUANTUM,
                        sample_rate=q_sample_rate)

        #TODO: FORCE LOOP TO BE LEAF IN ITSELF?

        if self._awg._amplitude_offset_handling == AWGAmplitudeOffsetHandling.IGNORE_OFFSET:
            voltage_offsets = (0.,) * self._awg.num_channels
        elif self._awg._amplitude_offset_handling == AWGAmplitudeOffsetHandling.CONSIDER_OFFSET:
            voltage_offsets = self._awg.offsets()
        else:
            raise ValueError('{} is invalid as AWGAmplitudeOffsetHandling'.format(self._amplitude_offset_handling))

        amplitudes = self._awg.amplitudes()
        
        helper_program_entry = HDAWGProgramEntry(loop,
                                                 selection_index=0,
                                                 waveform_memory=WaveformMemory(),
                                                 program_name="dummy_entry",
                                                 #TODO: handle channels, marks, voltage_trafos correctly
                                                 channels=tuple(playback_ids),
                                                 markers=tuple(marker_ids),
                                                 voltage_transformations=tuple(voltage_trafos),
                                                 amplitudes=amplitudes,
                                                 offsets=voltage_offsets,
                                                 sample_rate=q_sample_rate,
                                                 command_tables=('','','','')
                                                 )
        
        wf = ConcatenatedWaveform()
        
        #should be ordered (?)
        for waveform, bin_tuple in helper_program_entry._waveforms.items():
            wf.append(bin_tuple)
            
        wf.finalize()
        
        return wf.as_binary()
 
    
class SimpleChargeScanProgram(FixedStructureProgram):
    
    '''
    parameters:
    t_point: float,
    start_amp_1,end_amp_1,n_points_1,
    start_amp_2,end_amp_2,n_points_2,
    '''
    
    def __init__(self,
                 name: str,
                 parameters: dict,
                 channels_fast: tuple,
                 channels_slow: tuple,
                 measurement_channels: List[str], #["A",...]
                 awg_object: HDAWGChannelGroup,
                 dac_object: AlazarCard,
                 hardware_setup: HardwareSetup,
                 inner_pt: PulseTemplate = None, #must be defined on all channels (could be made better)
                 non_looped_constant_vals: dict = {},
                 point_measurement_window: tuple = (0.1,0.9), #fractions of measurement mask at each data point
                 auto_register: bool = True,
                 #TODO: measurements
                 ):
        
        self.parameters_scsp = parameters
        self._inner_pt = inner_pt
        self._non_looped_constant_vals = non_looped_constant_vals
        self._channels_scsp = {"fast":channels_fast,"slow":channels_slow}
        self._awg = awg_object
        self._dac = dac_object
        self._point_measurement_window = point_measurement_window
        
        self._make_pt()
        self._table_commands = {}
        
        self.always_reinstantiate = True #safer?
        self.binary_wf_tuple, self.inner_wave_names = None, {}
        
        super().__init__(name,*self.corresponding_pt,
                         measurement_channels=measurement_channels,
                         awg_object=awg_object,hardware_setup=hardware_setup,dac_object=dac_object,
                         auto_register=auto_register,)#TODO: measurements)
     
       
    @property
    def corresponding_pt(self) -> Tuple[PulseTemplate,dict]:
        return self._corresponding_pt, self.parameters_scsp
    
    def _make_pt(self):
        #TODO
        #slow_pt = PointPT()
        #TODO: why 0.5???
        #TODO: perhaps not jump but hold?
        ppt = PointPT([(0,"start_amp+i/n_steps*(end_amp-start_amp)",'jump'),("t_point","start_amp+i/n_steps*(end_amp-start_amp)",'jump'),],channel_names=["ch"],identifier="scsp_loop_step",
                      
                      )
        # ppt2 = PointPT([(0,"start_amp",'jump'),("1e9*t_init","start_amp",'jump'),],channel_names=["ch"],identifier="scsp_init_pt")

        slow_pts = tuple([(ppt,dict(i='i',n_steps=max(1,self.parameters_scsp['n_points_slow']-1),start_amp=self.parameters_scsp['start_amp_slow'],end_amp=self.parameters_scsp['end_amp_slow']),dict(ch=chi)) for chi in self._channels_scsp['slow']])
        fast_pts = tuple([(ppt,dict(i='j',n_steps=max(1,self.parameters_scsp['n_points_fast']-1),start_amp=self.parameters_scsp['start_amp_fast'],end_amp=self.parameters_scsp['end_amp_fast']),dict(ch=chi)) for chi in self._channels_scsp['fast']])
        
        remaining_pts = []
        loop_chs = self._channels_scsp['fast']+self._channels_scsp['slow']
        # t_init_before = super().NEW_MIN_QUANT/self._awg.sample_rate
        for i, chi in enumerate(super().CHANNELSTRING):
            if chi not in loop_chs:
                remaining_pts.append((ppt,dict(i=0,n_steps=1,start_amp=self._non_looped_constant_vals.get(chi,0),end_amp=self._non_looped_constant_vals.get(chi,0)),dict(ch=chi)))
        
        
        ppt_multi_channel = AtomicMultiChannelPT(*slow_pts,*fast_pts,*remaining_pts,
                                            identifier="SimpleChargeScanProgram_loop",
                                            
                                            )
        
        inner_time = 0
        if self._inner_pt is not None:
            ppt_multi_channel = SequencePT(self._inner_pt,ppt_multi_channel)
            inner_time = self._inner_pt.duration.evaluate_in_scope(self.parameters_scsp)
        
        self._n_measurements = self.parameters_scsp['n_points_slow']*self.parameters_scsp['n_points_fast'] if self._dac is not None else 0
        
        measurements=[(self.MEASUREMENT_NAME_TEMPLATE.format(i=i),
                        (i+self._point_measurement_window[0])*self.parameters_scsp['t_point']+(i+1)*inner_time,
                        (self._point_measurement_window[1]-self._point_measurement_window[0])*self.parameters_scsp['t_point']) for i in range(self._n_measurements)]
        
        # measurements=[(self.MEASUREMENT_NAME_TEMPLATE.format(i=i),
        #                 0,
        #                 100*self.parameters_scsp['t_point']) for i in range(1)]
        
        self._corresponding_pt = ForLoopPT(ForLoopPT(ppt_multi_channel,'j',(0,self.parameters_scsp['n_points_slow'],1)),'i',(0,self.parameters_scsp['n_points_fast'],1),
                                           measurements=measurements,
                                           # ,{'M':self.MEASUREMENT_NAME_TEMPLATE.format())}
                                           )
        
        self._scs_measurement_definitions = measurements
        
        # init after all not necessary?
        # init_pts = []
        # for i, chi in enumerate(self.CHANNELSTRING):
        #     init_pts.append((ppt2,dict(t_init=t_init_before,start_amp=self._non_looped_constant_vals.get(chi,0)),dict(ch=chi)))
            
        # init_pt_multi_channel = AtomicMultiChannelPT(*init_pts,identifier="SimpleChargeScanProgram_init")
            
        # self._corresponding_pt = SequencePT(init_pt_multi_channel,self._corresponding_pt)
        
        return
        
        
    
    def get_seqc_program_body(self, pos_var_name: str, indent="  ") -> Tuple[str,int]:
    
        ct_idx = pos_var_name
 
        if self._inner_pt is not None:
            inner_wf_string = f"executeTableEntry({ct_idx});"
            ct_add = '+1'
        else:
            inner_wf_string = ""
            ct_add = ''
        
        n_points_1 = self.parameters_scsp['n_points_slow'] - 1
        n_points_2 = self.parameters_scsp['n_points_fast'] - 1
        
        seqc_charge_scan = f"""\
         {inner_wf_string}
         executeTableEntry({ct_idx}{ct_add});
         executeTableEntry({ct_idx}{ct_add}+2);
         repeat({n_points_1}) {{
             repeat({n_points_2}){{
                 {inner_wf_string}
                 executeTableEntry({ct_idx}{ct_add}+1);
                 executeTableEntry({ct_idx}{ct_add}+2);
             }}
             {inner_wf_string}
             executeTableEntry({ct_idx}{ct_add}+3);
             executeTableEntry({ct_idx}{ct_add}+2);
         }}
         repeat({n_points_2}) {{ {inner_wf_string}executeTableEntry({ct_idx}{ct_add}+1);executeTableEntry({ct_idx}{ct_add}+2);}}
        
        """
        return textwrap.indent(textwrap.dedent(seqc_charge_scan),indent)
    
    def _compile_bwf(self):
        self.binary_wf_tuple = self.pt_to_binaries(self._inner_pt,self.parameters_scsp)
        self.inner_wave_names = ["scsp_"+str(id(x)) for x in self.binary_wf_tuple]
    
    def wf_definitions_iter(self) -> Iterator[Tuple[str,str]]:
                
        if self.binary_wf_tuple is None and self._inner_pt is not None:
            self._compile_bwf()
        
        f_counter = 0
        for i,wf_name in enumerate(self.inner_wave_names):
            f_name = self.FILE_NAME_TEMPLATE.format(hash=wf_name.replace('.csv','')+'_'+str(f_counter))
            f_counter += 1
            yield f_name, self.binary_wf_tuple[i]
    
    
    def wf_declarations_and_ct(self,ct_tuple,ct_start_index,wf_start_index,waveforms_tuple):
        
        ct_idx, wf_idx = ct_start_index, wf_start_index
        
        def round_to_multiple(number, multiple):
            return multiple * round(number / multiple)
        
        loop_chs = self._channels_scsp['fast']+self._channels_scsp['slow']

        # remaining_vals = {}
        
        wf_decl_string = []
        
        #TODO: implement non-zero volts on other channels
        f_counter = 0
        if self._inner_pt is not None:
            if self.binary_wf_tuple is None:
                self._compile_bwf()
                
            # inner_wave_str = ','.join(self.inner_wave_names)
            for i,bwf in enumerate(self.binary_wf_tuple):
                file_name = self.FILE_NAME_TEMPLATE.format(hash=self.inner_wave_names[i].replace('.csv','')+'_'+str(f_counter))
                
                
                
                # wf_decl_string.append(f'wave {self.inner_wave_names[i]} = "{file_name}";')
                
                f_counter += 1
                # waveforms_tuple[i][wf_idx] = (bwf.ch1,bwf.ch2,bwf.marker_data)
                waveforms_tuple[i][wf_idx] = (bwf.ch1,bwf.ch2,(0b1111*np.ones(len(bwf.ch1))).astype(int))
                #!!! force a marker here to have correct triggering....
                
                
                # inner_wave_str = ','.join(self.inner_wave_names)
            # #TODO: if this does not work, split up assignments to 4 different ones -> 4 different ct versions
                # wf_decl_string.append('assignWaveIndex({wave_str},{index});'.format(index=wf_idx, wave_str=inner_wave_str))
                
            inner_wave_str = ','.join(['placeholder({length},true,true)'.format(length=len(self.binary_wf_tuple[0].data)//3)]*8) #should be same for all
            wf_decl_string.append('assignWaveIndex({wave_str},{index});'.format(index=wf_idx, wave_str=inner_wave_str))
                # inner_wave_str = self.inner_wave_names[i]
                # wf_decl_string.append('assignWaveIndex({wave_str},{index});'.format(index=wf_idx+i, wave_str=inner_wave_str))
            
            for i in range(len(self.CHANNELSTRING)//2):
                # ct_tuple[i].table[ct_idx].waveform.index = wf_idx+i
                ct_tuple[i].table[ct_idx].waveform.index = wf_idx
                ct_tuple[i].table[ct_idx].amplitude0.value = 1.0
                ct_tuple[i].table[ct_idx].amplitude0.increment = False
                ct_tuple[i].table[ct_idx].amplitude0.register = 0
                ct_tuple[i].table[ct_idx].amplitude1.value = 1.0
                ct_tuple[i].table[ct_idx].amplitude1.register = 0
                ct_tuple[i].table[ct_idx].amplitude1.increment = False
            
            ct_idx += 1
            # wf_idx += len(self.CHANNELSTRING)//2
            wf_idx += 1

            
        # wf_decl_string.append(f'wave scs = "{file_name}";')
        
        amplitude_divisor_tuple = self._awg.amplitudes()
        
        start_amp_slow = self.parameters_scsp['start_amp_slow']
        start_amp_fast = self.parameters_scsp['start_amp_fast']
        end_amp_slow = self.parameters_scsp['end_amp_slow']
        end_amp_fast = self.parameters_scsp['end_amp_fast']
        n_points_slow = self.parameters_scsp['n_points_slow'] - 1
        n_points_fast = self.parameters_scsp['n_points_fast'] - 1
        
        #TODO: not implicit
        N_hold = round_to_multiple(max(self.NEW_MIN_QUANT,int(1e-9*self.parameters_scsp['t_point']*self._awg.sample_rate-self.NEW_MIN_QUANT)),self._awg.WAVEFORM_LEN_QUANTUM)
        delta_amp_slow, delta_amp_fast = (end_amp_slow-start_amp_slow)/n_points_slow, (end_amp_fast-start_amp_fast)/n_points_fast
        
        def get_mod2_amp(ct,i):
            if i%2==0:
                return ct.amplitude0
            else:
                return ct.amplitude1
        
        #to circumvent error upon defining multiple indices for ones(). doesnt make sense why it is necessary, but apparently is.
        #TODO: doesn't work. apparently still defined as some internal reference to original wf and then cannot assign second index... annoying
        t_code = int(time.time())
        quant=self.NEW_MIN_QUANT
        # wf_decl_string.append(f'wave ones_{t_code} = ones({quant});')
        # wf_decl_string.append('assignWaveIndex(ones_{t_code},ones_{t_code},{index});'.format(index=wf_idx,t_code=t_code))
        # ones_dcl_str = ','.join([f'{i},ones_{t_code}' for i in range(1,9)])
        # ones_dcl_str = ','.join([f'{i},ones_{t_code}' for i in range(1,9)])
        ones_dcl_str = ','.join([f'placeholder({quant},true,true)']*8) #should be same for all
        wf_decl_string.append(f'assignWaveIndex({ones_dcl_str},{wf_idx});')
        
        for i in range(4):
            waveforms_tuple[i][wf_idx] = (np.ones(quant),np.ones(quant),(0b1111*np.ones(quant)).astype(int))

        
        self._last_ct_loop_start_idx = ct_idx

        # wf_decl = ''
        # loop_string = ''
        #TODO: ensure that default amp register =0 is untouched?
        for i,ch in enumerate(self.CHANNELSTRING):
            
            #playHold
            ct_tuple[i//2].table[ct_idx+2].waveform.playHold = True
            ct_tuple[i//2].table[ct_idx+2].waveform.length = N_hold
            
            ct_tuple[i//2].table[ct_idx].waveform.index = wf_idx
            ct_tuple[i//2].table[ct_idx+1].waveform.index = wf_idx
            ct_tuple[i//2].table[ct_idx+3].waveform.index = wf_idx
            
            if ch not in loop_chs:
                
                # wf_decl += 'ones'
                
                #init of channel before loop
                get_mod2_amp(ct_tuple[i//2].table[ct_idx],i).value = self._non_looped_constant_vals.get(ch,0) / amplitude_divisor_tuple[i]
                get_mod2_amp(ct_tuple[i//2].table[ct_idx],i).register = 1
                get_mod2_amp(ct_tuple[i//2].table[ct_idx],i).increment = False

                #inner loop increment
                get_mod2_amp(ct_tuple[i//2].table[ct_idx+1],i).value = self._non_looped_constant_vals.get(ch,0) / amplitude_divisor_tuple[i]
                get_mod2_amp(ct_tuple[i//2].table[ct_idx+1],i).increment = False
                get_mod2_amp(ct_tuple[i//2].table[ct_idx+1],i).register = 1
                
                #outer loop increment
                get_mod2_amp(ct_tuple[i//2].table[ct_idx+3],i).value = self._non_looped_constant_vals.get(ch,0) / amplitude_divisor_tuple[i]
                get_mod2_amp(ct_tuple[i//2].table[ct_idx+3],i).increment = False
                get_mod2_amp(ct_tuple[i//2].table[ct_idx+3],i).register = 1
                
                
            elif ch in self._channels_scsp['slow']:
                
                
                #init of channel before loop
                get_mod2_amp(ct_tuple[i//2].table[ct_idx],i).value = start_amp_slow / amplitude_divisor_tuple[i]
                get_mod2_amp(ct_tuple[i//2].table[ct_idx],i).register = 1
                get_mod2_amp(ct_tuple[i//2].table[ct_idx],i).increment = False

                #inner loop increment
                get_mod2_amp(ct_tuple[i//2].table[ct_idx+1],i).value = 0.0
                get_mod2_amp(ct_tuple[i//2].table[ct_idx+1],i).increment = True
                get_mod2_amp(ct_tuple[i//2].table[ct_idx+1],i).register = 1
                
                #outer loop increment
                get_mod2_amp(ct_tuple[i//2].table[ct_idx+3],i).value = delta_amp_slow / amplitude_divisor_tuple[i]
                get_mod2_amp(ct_tuple[i//2].table[ct_idx+3],i).increment = True
                get_mod2_amp(ct_tuple[i//2].table[ct_idx+3],i).register = 1


            elif ch in self._channels_scsp['fast']:
                
                #init of channel before loop
                get_mod2_amp(ct_tuple[i//2].table[ct_idx],i).value = start_amp_fast / amplitude_divisor_tuple[i]
                get_mod2_amp(ct_tuple[i//2].table[ct_idx],i).register = 1
                get_mod2_amp(ct_tuple[i//2].table[ct_idx],i).increment = False

                
                #inner loop increment
                get_mod2_amp(ct_tuple[i//2].table[ct_idx+1],i).value = delta_amp_fast / amplitude_divisor_tuple[i]
                get_mod2_amp(ct_tuple[i//2].table[ct_idx+1],i).increment = True
                get_mod2_amp(ct_tuple[i//2].table[ct_idx+1],i).register = 1

                
                #outer loop increment
                get_mod2_amp(ct_tuple[i//2].table[ct_idx+3],i).value = start_amp_fast / amplitude_divisor_tuple[i]
                get_mod2_amp(ct_tuple[i//2].table[ct_idx+3],i).increment = False
                get_mod2_amp(ct_tuple[i//2].table[ct_idx+3],i).register = 1

            else:
                raise RuntimeError
   
        self._ct_tuple_reference = ct_tuple
        
        return '\n'.join(wf_decl_string), ct_idx+4, wf_idx+1


    def update_params_in_place(self,new_looped_params: Dict[str,float],
                               new_non_looped_constant_vals: Dict[str,float],
                               ) -> None:
        
        
        loop_chs = self._channels_scsp['fast']+self._channels_scsp['slow']
        
        assert_no_key_in_list(new_non_looped_constant_vals,loop_chs)
        
        self.parameters_scsp.update(remove_unallowed_keys(new_looped_params,("start_amp_slow","start_amp_fast","end_amp_slow","end_amp_fast")))
        self._non_looped_constant_vals.update(remove_unallowed_keys(new_non_looped_constant_vals,[char for char in self.CHANNELSTRING]))
        #probably only works if no other programs have been added in the meantime?
        #or rather if none has been removed in order before?
        # in any case, sequence must be on device before this is called
        
        self._awg.arm(None)
        self._awg.enable(False)
        
        ct_tuple = self._ct_tuple_reference
        ct_idx = self._last_ct_loop_start_idx
        
        amplitude_divisor_tuple = self._awg.amplitudes()
        
        start_amp_slow = self.parameters_scsp['start_amp_slow']
        start_amp_fast = self.parameters_scsp['start_amp_fast']
        end_amp_slow = self.parameters_scsp['end_amp_slow']
        end_amp_fast = self.parameters_scsp['end_amp_fast']
        
        n_points_slow = self.parameters_scsp['n_points_slow'] - 1
        n_points_fast = self.parameters_scsp['n_points_fast'] - 1
        
        #TODO: not implicit
        delta_amp_slow, delta_amp_fast = (end_amp_slow-start_amp_slow)/n_points_slow, (end_amp_fast-start_amp_fast)/n_points_fast
        
        for i,ch in enumerate(self.CHANNELSTRING):
            
            if ch not in loop_chs:
                                
                #init of channel before loop
                get_mod2_amp(ct_tuple[i//2].table[ct_idx],i).value = self._non_looped_constant_vals.get(ch,0) / amplitude_divisor_tuple[i]

                #inner loop increment
                get_mod2_amp(ct_tuple[i//2].table[ct_idx+1],i).value = self._non_looped_constant_vals.get(ch,0) / amplitude_divisor_tuple[i]

                #outer loop increment
                get_mod2_amp(ct_tuple[i//2].table[ct_idx+3],i).value = self._non_looped_constant_vals.get(ch,0) / amplitude_divisor_tuple[i]
                
                
            elif ch in self._channels_scsp['slow']:
                
                
                #init of channel before loop
                get_mod2_amp(ct_tuple[i//2].table[ct_idx],i).value = start_amp_slow / amplitude_divisor_tuple[i]

                #inner loop increment
                get_mod2_amp(ct_tuple[i//2].table[ct_idx+1],i).value = 0.0
     
                #outer loop increment
                get_mod2_amp(ct_tuple[i//2].table[ct_idx+3],i).value = delta_amp_slow / amplitude_divisor_tuple[i]


            elif ch in self._channels_scsp['fast']:
                
                #init of channel before loop
                get_mod2_amp(ct_tuple[i//2].table[ct_idx],i).value = start_amp_fast / amplitude_divisor_tuple[i]
         
                #inner loop increment
                get_mod2_amp(ct_tuple[i//2].table[ct_idx+1],i).value = delta_amp_fast / amplitude_divisor_tuple[i]
        
                #outer loop increment
                get_mod2_amp(ct_tuple[i//2].table[ct_idx+3],i).value = start_amp_fast / amplitude_divisor_tuple[i]

            else:
                raise RuntimeError
        
        self._awg._upload_ct_tuple(tuple([json.dumps(ct.as_dict()) for ct in ct_tuple]))
        self._awg.arm(self.name)
        
        return
        

    def waveform_dict(self) -> dict:
        if self.binary_wf_tuple is None and self._inner_pt is not None:
            self._compile_bwf()
        
        #TODO: originally intended as "Waveform" key, but did not trace correct definitions yet
        return {0: self.binary_wf_tuple}
    
    
    #here not necessary to employ? Seems to be.
    def expand_ct(self,
            ct: CommandTable,
            starting_index: int
            ) -> CommandTable:
        
        assert self._seqc_body is not None
        
        return ct
    
    def get_2d_data_dict(self) -> np.ndarray:
        
        voltages = {}
        if self._measurement_result is None:
            return voltages
        
        # for i,ch in enumerate(self._measurement_channels):
        #     voltages[ch] = np.concatenate([self.measurement_result[f"M{index}{ch}"] for index in range(self._n_measurements)]).reshape(
        #         (self.parameters_scsp['n_points_slow'],self.parameters_scsp['n_points_fast']),order='C')
        for i,ch in enumerate(self._measurement_channels):
            voltages[ch] = np.array([np.mean(self.measurement_result[f"M{index}{ch}"]) for index in range(self._n_measurements)]).reshape(
                (self.parameters_scsp['n_points_slow'],self.parameters_scsp['n_points_fast']),order='C')
        
        return voltages
    
    def plot_all_measurements(self):
        n_plots = len(self._measurement_channels)
        
        fig = plt.figure(figsize=(3.5*n_plots,3.0),dpi=300)

        gs0 = fig.add_gridspec(n_plots, 1, width_ratios=[1,]*n_plots)
        
        data_dict = self.get_2d_data_dict()
        
        #TODO
        # fig, ax    
        #plot
        return



def get_mod2_amp(ct,i):
    if i%2==0:
        return ct.amplitude0
    else:
        return ct.amplitude1
    
def remove_unallowed_keys(dictionary, allowed_keys):
    removed_keys = []
    for key in list(dictionary.keys()):
        if key not in allowed_keys:
            dictionary.pop(key)
            removed_keys.append(key)
            print(f"Warning: Key '{key}' cannot be updated.")
    return dictionary

def assert_no_key_in_list(dictionary, strings):
    for key in dictionary.keys():
        assert key not in strings, f"Channel '{key}' is a looped channel."


