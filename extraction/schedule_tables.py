#%%
# -*- coding: utf-8 -*-
import os
import subprocess
from lxml import etree
from lxml.builder import ElementMaker # lxml only !
import datetime
from support_modules import support as sup


class TimeTablesCreator():
    '''
        This class creates the resources timetables and associates them
        to the resource pools
     '''

    def __init__(self, settings):
        '''constructor'''
        self.settings = settings


    def create_timetables(self, calendar_method):
        creator = self._get_creator(calendar_method)
        sup.print_performed_task('Mining calendars')
        return creator()

    def _get_creator(self, calendar_method):
        if calendar_method == 'default':
            return self._default_creator
        elif calendar_method == 'discovery':
            return self._discovery_creator
        else:
            raise ValueError(calendar_method)

    def _default_creator(self) -> None:
        """
        Creates predefined timetables for BIMP simulator
        """
        time_table = list()
        if self.settings['dtype'] == 'LV917':
            time_table.append({'id_t': 'QBP_DEFAULT_TIMETABLE',
                                    'default': 'true',
                                    'name': 'Default',
                                    'from_t': '09:00:00.000+00:00',
                                    'to_t': '17:00:00.000+00:00',
                                    'from_w': 'MONDAY',
                                    'to_w': 'FRIDAY'})
        elif self.settings['dtype'] == '247':
            time_table.append({'id_t': 'QBP_DEFAULT_TIMETABLE',
                                    'default': 'true',
                                    'name': '24/7',
                                    'from_t': '00:00:00.000+00:00',
                                    'to_t': '23:59:59.999+00:00',
                                    'from_w': 'MONDAY',
                                    'to_w': 'SUNDAY'})
        if self.settings['simulator'] == 'bimp':
            time_table = self._print_xml_bimp(time_table)
        self.time_table = time_table
        self.time_table_name = 'QBP_DEFAULT_TIMETABLE'
        sup.print_done_task()
    
    def _discovery_creator(self) -> None:
        """Executes BIMP Simulations.
        Args:
            settings (dict): Path to jar and file names
            rep (int): repetition number
        """
        args = ['java', '-jar', self.settings['calender_path'],
                os.path.join(self.settings['input'], self.settings['file']),
                str(self.settings['support']), str(self.settings['confidence'])]
        process = subprocess.run(args, check=True, stdout=subprocess.PIPE)
        found = False
        xml = ['<qbp:timetables xmlns:qbp="http://www.qbp-simulator.com/Schema201212">']
        for line in process.stdout.decode('utf-8').splitlines():
            if 'BIMP Calendar' in line:
                found = False if found else True
            if found and not 'BIMP Calendar' in line:
                xml.append(line.strip())
        xml.append('</qbp:timetables>')
        xml = etree.fromstring(''.join(xml).strip())
        ns = {'qbp': "http://www.qbp-simulator.com/Schema201212"}
        # Fix timestamp format
        rules = (xml.find('qbp:timetable', namespaces=ns)
                 .find('qbp:rules', namespaces=ns)
                 .findall('qbp:rule', namespaces=ns))
        for rule in rules:
           rule.attrib['fromTime'] = (
               datetime.datetime.strptime(rule.attrib['fromTime'], "%H:%M")
               .strftime("%H:%M:%S.000+00:00"))
           rule.attrib['toTime'] = (
               datetime.datetime.strptime(rule.attrib['toTime'], "%H:%M")
               .strftime("%H:%M:%S.000+00:00"))
        # Sava values
        self.time_table = xml
        self.time_table_name = (xml.find(
            'qbp:timetable', namespaces=ns).attrib['id'])
        sup.print_done_task()

    @staticmethod
    def _print_xml_bimp(time_table) -> str:
        E = ElementMaker(namespace='http://www.qbp-simulator.com/Schema201212',
                         nsmap={'qbp':
                                "http://www.qbp-simulator.com/Schema201212"})
        TIMETABLES = E.timetables
        TIMETABLE = E.timetable
        RULES = E.rules
        RULE = E.rule
        my_doc = TIMETABLES(
			*[
				TIMETABLE(
					RULES(
                        RULE(fromTime=table['from_t'],
                             toTime=table['to_t'],
                             fromWeekDay=table['from_w'],
                             toWeekDay=table['to_w'])),
                    id=table['id_t'],
                    default=table['default'],
                    name=table['name']
				) for table in time_table
			]
		)
        return my_doc
