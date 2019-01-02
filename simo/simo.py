# -*- coding: utf-8 -*-
from extraction import parameter_extraction as par
from readers import readers as rd
from writers import xml_writer as xml_bimp
from writers import xml_writer_scylla as xml_scylla
from analyzers import analyzer as az

import sys
import support as sup
import configparser as cp
import subprocess
import os
import platform as pl

def orchestra():
    """SIMO orchestrate method"""
    Config = cp.ConfigParser(interpolation=None)
    Config.read("./config.ini")
    #File name with extension
    file = Config.get('FILE', 'name')
    #File name array
    file_name = str.split(file, ".")
    log_file_name = Config.get('FILES', 'input') + file
    bpmn_file_name = Config.get('FILES', 'input') + file_name[0] + '.bpmn'
    #Folder of the output bimp simulator
    output_file_bimp = Config.get('FILES', 'output_file_bimp') + file_name[0] + '.bpmn'
    # Steps EXECUTION
    mining = Config.get('EXECUTION','mining') in ['true', 'True', '1', 'Yes', 'yes']
    alignment = Config.get('EXECUTION','alignment') in ['true', 'True', '1', 'Yes', 'yes']
    parameters = Config.get('EXECUTION','parameters') in ['true', 'True', '1', 'Yes', 'yes']
    simulation = Config.get('EXECUTION','simulation') in ['true', 'True', '1', 'Yes', 'yes']
    analysis = Config.get('EXECUTION','analysis') in ['true', 'True', '1', 'Yes', 'yes']
    sim_cycles = int(Config.get('EXECUTION','simcycles'))
    #User selection of the miner {SplitMiner, Prom}
    miner = Config.get('MINING', 'miner')
    splitMiner = Config.get('MINING', 'splitMiner')
    splitMOutput = Config.get('FILES', 'input') + file_name[0]
    promMiner = Config.get('MINING', 'promMiner')
    promInput1 = Config.get('MINING', 'promInput1')
    promInput2 = Config.get('MINING', 'promInput2')
    alignment_input = Config.get('FILES', 'input')
    # User selection of the simulator {Bimp, Scylla}
    simulator = Config.get('SIMULATOR', 'simulator')
    bothSimulators = Config.get('SIMULATOR', 'bothSimulators') in ['true', 'True', '1', 'Yes', 'yes']
    manual_model = Config.get('SIMULATOR', 'manualModel') in ['true', 'True', '1', 'Yes', 'yes']
    manual_sim_model_file = Config.get('SIMULATOR', 'manualSimModelFile')
    manual_model_file = Config.get('SIMULATOR', 'manualModelFile')
    simBimp = Config.get('SIMULATOR', 'simBimp')
    simBimpParam2 = Config.get('SIMULATOR', 'simBimpParam2')
    simBimpParam3 = Config.get('SIMULATOR', 'simBimpParam3')
    scylla = Config.get('SIMULATOR', 'scylla')
    scyllaParam1 = Config.get('SIMULATOR', 'scyllaParam1')
    #Step 1
    if mining:
        print(" -- Mining Process Structure --")
        #Check if split miner was a choice of the user
        if ('splitMiner' in miner):
            """Call split Miner Jar
                Param 1: Path to jar
                Param 2: Path to log process
                Param 3: Path to splitMiner output"""
            subprocess.call(
                ['java', '-jar', splitMiner, log_file_name, splitMOutput])
        # Check if prom was a choice of the user
        if ('prom' in miner):
            """Call Prom Miner Jar
                Param 1: Path to jar
                Param 2: -f command
                Param 3: Path to promM execution script"""
            subprocess.call(
                ['java', '-jar', promMiner, promInput1, promInput2]
            )
    #Step 2
    if alignment:
        print("-- Process Alignment --")
        proConformance = Config.get('PROCONFORMANCE', 'proconformance')
        # print(proConformance)
        subprocess.call(
         	['java','-jar',proConformance, alignment_input, file, file_name[0] + '.bpmn', "true"], bufsize=-1)
    #Step 3
    if parameters:
        print("-- Mining Simulation Parameters --")
        end_timeformat = Config.get('EXECUTION','endtimeformat')
        start_timeformat = Config.get('EXECUTION','starttimeformat')
        log_columns_numbers = sup.reduce_list(Config.get('EXECUTION','logcolumnsnumbers'))
        log, bpmn = rd.read_inputs(start_timeformat, end_timeformat, log_columns_numbers, log_file_name, bpmn_file_name)

        if alignment:
            align_info_file = Config['EXECUTION']['aligninfofile']
            align_type_file = Config['EXECUTION']['aligntypefile']
            if(bothSimulators):
                parameters, process_stats,parameters_scylla = par.extract_parameters(log, bpmn, True, True, align_info_file,
                                                                      align_type_file)
                scylla_folder= scyllaParam1 + sup.folder_id()
                if not os.path.exists(scylla_folder):
                  os.makedirs(scylla_folder)
                xml_bimp.print_parameters(bpmn_file_name, output_file_bimp, parameters)
                xml_scylla.print_parameters(bpmn_file_name, os.path.join(scylla_folder,file_name[0] + '.xml'),
                                            parameters_scylla)
            elif ('bimp' in simulator):
                parameters,process_stats,_ = par.extract_parameters(log, bpmn, True,False, align_info_file, align_type_file)
                xml_bimp.print_parameters(bpmn_file_name, output_file_bimp, parameters)
            # Check if Scylla was a choice of the user
            elif ('scylla' in simulator):
                scylla_folder= scyllaParam1 + sup.folder_id()
                if not os.path.exists(scylla_folder):
                    os.makedirs(scylla_folder)
                # par.extract_parameters(log, bpmn, splitMOutput + '.bpmn', '../sim/' + file_name[0] + '.xml', assets_output,False)
                _,process_stats,parameters_scylla = par.extract_parameters(log, bpmn, False,True, align_info_file, align_type_file)
                xml_scylla.print_parameters(bpmn_file_name, os.path.join(scylla_folder,file_name[0] + '.xml'),parameters_scylla)
        else:
            if (bothSimulators):
                parameters, process_stats, parameters_scylla  = par.extract_parameters(log, bpmn, True, True)
                scylla_folder = scyllaParam1 + sup.folder_id()
                if not os.path.exists(scylla_folder):
                  os.makedirs(scylla_folder)
                xml_bimp.print_parameters(bpmn_file_name, output_file_bimp, parameters)
                xml_scylla.print_parameters(bpmn_file_name, os.path.join(scylla_folder,file_name[0] + '.xml'),
                                            parameters_scylla)
            elif ('bimp' in simulator):
                parameters,process_stats,_  = par.extract_parameters(log, bpmn, True, False)
                xml_bimp.print_parameters(bpmn_file_name, output_file_bimp, parameters)
            # Check if Scylla was a choice of the user
            elif ('scylla' in simulator):
                scylla_folder= scyllaParam1 + sup.folder_id()
                if not os.path.exists(scylla_folder):
                  os.makedirs(scylla_folder)
                _,process_stats,parameters_scylla = par.extract_parameters(log, bpmn, False, True)
                xml_scylla.print_parameters(bpmn_file_name, os.path.join(scylla_folder,file_name[0] + '.xml'),parameters_scylla)

    #Check if Bimp was a choice of the user
    if simulation:
        if ('bimp' in simulator or bothSimulators):
            print("-- Executing BIMP Simulations --")
            folder=sup.folder_id()
            if not os.path.exists(simBimpParam3 + folder):
                os.makedirs(simBimpParam3 + folder)
            for i in range(sim_cycles):
                print("Experiment #" + str(i + 1))
                #Param 1: Path to jar
                #Param 3: SiMo BPMN file generated
                #Param 4: -csv output format
                #Param 5: Output folder and file
                subprocess.call(
                    ['java', '-jar', simBimp, output_file_bimp, simBimpParam2, os.path.join(simBimpParam3 + folder , str(i + 1) + ".csv")]
                )
            bimp_statistics = rd.import_bimp_statistics(simBimpParam3 + folder, bpmn_file_name)
        # Check if Scylla was a choice of the user
        if ('scylla' in simulator or bothSimulators):
            source = os.path.join("../inputs" ,file_name[0] + '.bpmn')
            destiny = os.path.join(scylla_folder , file_name[0] + '.bpmn')
            if pl.system().lower() == 'windows':
                os.system('copy "' + source + '" "' + destiny + '"')
            else:
                os.system('cp "' + source + '" "' + destiny + '"')
            print("-- Executing Scylla Simulations --")
            # Param 1: Path to jar
            # Param 3: Folder where the files are located
            # Param 4: Config xml file
            # Param 5: Bpmn process file
            # Param 6: SiMo Output simulation file.
            scylla_statistics = list()
            for i in range(sim_cycles):
                subprocess.call(
                    ['java', '-jar', scylla, scylla_folder, file_name[0] + 'conf.xml', file_name[0] + '.bpmn', file_name[0]+'simu.xml',str(i + 1)]
                )
                # Read xml output file from scylla
                scylla_statistics.extend(rd.import_scylla_statistics(os.path.join(scylla_folder ,'output'+str(i + 1)), file_name[0] + '.xes', bpmn_file_name, parameters_scylla, i))
        if manual_model:
            if ('bimp' in simulator or bothSimulators):
                print("-- Executing BIMP Manual Model Simulations --")
                folder=sup.folder_id()
                if not os.path.exists(simBimpParam3 + folder):
                    os.makedirs(simBimpParam3 + folder)
                for i in range(sim_cycles):
                    print("Experiment #" + str(i + 1))
                    #Param 1: Path to jar
                    #Param 3: SiMo BPMN file generated
                    #Param 4: -csv output format
                    #Param 5: Output folder and file
                    subprocess.call(
                        ['java', '-jar', simBimp, manual_sim_model_file, simBimpParam2, os.path.join(simBimpParam3 + folder , str(i + 1) + ".csv")]
                    )
                sim_manual_model = rd.import_bimp_statistics(simBimpParam3 + folder, manual_model_file,'manual-bimp')
    # Data analysis
    if analysis:
        print("-- Data Comparison --")
        if parameters and simulation:
            if 'scylla' in simulator or bothSimulators:
                process_stats.extend(scylla_statistics)
            if ('bimp' in simulator or bothSimulators):
                process_stats.extend(bimp_statistics)
            if manual_model:
                process_stats.extend(sim_manual_model)
        az.create_report(process_stats)

# --setup--
def main(argv):
    """Main aplication method"""
    orchestra()

if __name__ == "__main__":
    main(sys.argv[1:])
