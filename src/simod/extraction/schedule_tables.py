import datetime
import os
import subprocess
from pathlib import Path

import pandas as pd
from lxml import etree
from lxml.builder import ElementMaker  # lxml only !
from tqdm import tqdm

from .. import support_utils as sup
from ..cli_formatter import print_step
from ..configuration import Configuration, DataType, CalculationMethod, QBP_NAMESPACE_URI


class TimeTablesCreator:
    """This class creates the resources timetables and associates them to the resource pools"""

    def __init__(self, settings: Configuration):
        self.settings = settings

    def create_timetables(self, args):
        creator = self._get_creator(args['res_cal_met'], args['arr_cal_met'])
        if args['res_cal_met'] is CalculationMethod.POOL:
            return creator(args['resource_table'])
        else:
            return creator()

    def _get_creator(self, res_cal_met: CalculationMethod, arr_cal_met: CalculationMethod):
        if res_cal_met is CalculationMethod.DEFAULT and arr_cal_met is CalculationMethod.DEFAULT:
            return self._def_timetables
        elif res_cal_met is CalculationMethod.DEFAULT and arr_cal_met is CalculationMethod.DISCOVERED:
            return self._defres_disarr
        elif res_cal_met is CalculationMethod.DISCOVERED and arr_cal_met is CalculationMethod.DEFAULT:
            return self._disres_defarr
        elif res_cal_met is CalculationMethod.DISCOVERED and arr_cal_met is CalculationMethod.DISCOVERED:
            return self._disres_disarr
        elif res_cal_met is CalculationMethod.POOL and arr_cal_met is CalculationMethod.DEFAULT:
            return self._dispoolres_defarr
        elif res_cal_met is CalculationMethod.POOL and arr_cal_met is CalculationMethod.DISCOVERED:
            return self._dispoolres_disarr
        else:
            raise ValueError(res_cal_met, arr_cal_met)

    def _def_timetables(self) -> None:
        pbar = tqdm(total=2, desc='mining calendars:')
        xmlres = self._default_creator(self.settings.res_dtype, 1)
        pbar.update(1)
        xmlarr = self._default_creator(self.settings.arr_dtype, 2)
        pbar.update(1)

        # merge timetables
        ns = {'qbp': QBP_NAMESPACE_URI}
        restimetable = xmlres.find('qbp:timetable', namespaces=ns)
        arrivaltable = xmlarr.findall('qbp:timetable', namespaces=ns)
        index = len(arrivaltable)
        xmlarr.insert(index, restimetable)

        # save values
        self.time_table = xmlarr
        self.res_ttable_name = {'arrival': arrivaltable[0].attrib['id'],
                                'resources': restimetable.attrib['id']}

        pbar.close()

    def _defres_disarr(self) -> None:
        pbar = tqdm(total=2, desc='mining calendars:')
        xmlres = self._default_creator(self.settings.res_dtype, 1)
        pbar.update(1)
        xmlarr = self._timetable_discoverer(self.settings.calender_path, self.settings.log_path,
                                            str(self.settings.arr_support), str(self.settings.arr_confidence), 2)
        pbar.update(1)
        # merge timetables
        ns = {'qbp': QBP_NAMESPACE_URI}
        restimetable = xmlres.find('qbp:timetable', namespaces=ns)
        arrivaltable = xmlarr.findall('qbp:timetable', namespaces=ns)
        index = len(arrivaltable)
        xmlarr.insert(index, restimetable)

        # save values
        self.time_table = xmlarr
        self.res_ttable_name = {'arrival': arrivaltable[0].attrib['id'],
                                'resources': restimetable.attrib['id']}
        pbar.close()

    def _disres_defarr(self) -> None:
        pbar = tqdm(total=2, desc='mining calendars:')
        xmlres = self._timetable_discoverer(
            self.settings.calender_path,
            self.settings.log_path,
            str(self.settings.res_support),
            str(self.settings.res_confidence), 1)
        pbar.update(1)
        xmlarr = self._default_creator(self.settings.arr_dtype, 2)
        pbar.update(1)
        # merge timetables
        ns = {'qbp': QBP_NAMESPACE_URI}
        restimetable = xmlres.find('qbp:timetable', namespaces=ns)
        arrivaltable = xmlarr.findall('qbp:timetable', namespaces=ns)
        index = len(arrivaltable)
        xmlarr.insert(index, restimetable)

        # save values
        self.time_table = xmlarr
        self.res_ttable_name = {'arrival': arrivaltable[0].attrib['id'],
                                'resources': restimetable.attrib['id']}
        pbar.close()

    def _disres_disarr(self) -> None:
        pbar = tqdm(total=2, desc='mining calendars:')
        xmlres = self._timetable_discoverer(
            self.settings.calender_path,
            self.settings.log_path,
            str(self.settings.res_support),
            str(self.settings.res_confidence), 1)
        pbar.update(1)
        xmlarr = self._timetable_discoverer(
            self.settings.calender_path,
            self.settings.log_path,
            str(self.settings.arr_support),
            str(self.settings.arr_confidence), 2)
        pbar.update(1)
        # merge timetables
        ns = {'qbp': QBP_NAMESPACE_URI}
        restimetable = xmlres.find('qbp:timetable', namespaces=ns)
        arrivaltable = xmlarr.findall('qbp:timetable', namespaces=ns)
        index = len(arrivaltable)
        xmlarr.insert(index, restimetable)

        # save values
        self.time_table = xmlarr
        self.res_ttable_name = {'arrival': arrivaltable[0].attrib['id'], 'resources': restimetable.attrib['id']}
        pbar.close()

    def _dispoolres_defarr(self, timetable) -> None:
        pbar = tqdm(total=2, desc='mining calendars:')
        ns = {'qbp': QBP_NAMESPACE_URI}
        xmlarr = self._default_creator(self.settings.arr_dtype, 2)
        pbar.update(1)
        arrivaltable = xmlarr.find('qbp:timetable', namespaces=ns)

        tablenames = {'arrival': arrivaltable.attrib['id']}
        for k, group in pd.DataFrame(timetable).groupby('role'):
            temp_filename = os.path.join(self.settings.output, sup.file_id())
            group[['resource']].to_csv(temp_filename, index=False)
            xmlres = self._timetable_discoverer(
                self.settings.calender_path,
                self.settings.log_path,
                str(self.settings.res_support),
                str(self.settings.res_confidence), 3, temp_filename)
            restimetable = xmlres.find('qbp:timetable', namespaces=ns)
            restimetable.attrib['id'] = 'QBP_RES_' + k.upper().replace(' ', '_') + '_TIMETABLE'
            restimetable.attrib['name'] = k.upper().replace(' ', '_') + '_TIMETABLE'
            tablenames[k] = 'QBP_RES_' + k.upper().replace(' ', '_') + '_TIMETABLE'
            index = len(xmlarr.findall('qbp:timetable', namespaces=ns))
            xmlarr.insert(index, restimetable)
            os.unlink(temp_filename)
        pbar.update(1)
        # save values
        self.time_table = xmlarr
        self.res_ttable_name = tablenames
        pbar.close()

    def _dispoolres_disarr(self, timetable) -> None:
        pbar = tqdm(total=2, desc='mining calendars:')
        ns = {'qbp': QBP_NAMESPACE_URI}
        xmlarr = self._timetable_discoverer(
            self.settings.calender_path,
            self.settings.log_path,
            str(self.settings.arr_support),
            str(self.settings.arr_confidence), 2)
        pbar.update(1)
        arrivaltable = xmlarr.find('qbp:timetable', namespaces=ns)
        tablenames = {'arrival': arrivaltable.attrib['id']}
        for k, group in pd.DataFrame(timetable).groupby('role'):
            temp_filename = os.path.join(self.settings.output, sup.file_id())
            group[['resource']].to_csv(temp_filename, index=False)
            xmlres = self._timetable_discoverer(
                self.settings.calender_path,
                self.settings.log_path,
                str(self.settings.res_support),
                str(self.settings.res_confidence), 3, temp_filename)
            restimetable = xmlres.find('qbp:timetable', namespaces=ns)
            restimetable.attrib['id'] = 'QBP_RES_' + k.upper().replace(' ', '_') + '_TIMETABLE'
            restimetable.attrib['name'] = k.upper().replace(' ', '_') + '_TIMETABLE'
            tablenames[k] = 'QBP_RES_' + k.upper().replace(' ', '_') + '_TIMETABLE'
            index = len(xmlarr.findall('qbp:timetable', namespaces=ns))
            xmlarr.insert(index, restimetable)
            os.unlink(temp_filename)
        pbar.update(1)
        # save values
        self.time_table = xmlarr
        self.res_ttable_name = tablenames
        pbar.close()

    @staticmethod
    def _default_creator(dtype: DataType, mode) -> None:
        """
        Creates predefined timetables for BIMP simulator
        """

        def print_xml_bimp(time_table) -> str:
            E = ElementMaker(namespace=QBP_NAMESPACE_URI, nsmap={'qbp': QBP_NAMESPACE_URI})
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

        name = ('QBP_RES_DEFAULT_TIMETABLE'
                if mode == 1 else 'QBP_ARR_DEFAULT_TIMETABLE')
        default = 'false' if mode == 1 else 'true'
        time_table = list()
        if dtype == DataType.LV917:
            time_table.append({'id_t': name,
                               'default': default,
                               'name': 'Default',
                               'from_t': '09:00:00.000+00:00',
                               'to_t': '17:00:00.000+00:00',
                               'from_w': 'MONDAY',
                               'to_w': 'FRIDAY'})
        elif dtype == DataType.DT247:
            time_table.append({'id_t': name,
                               'default': default,
                               'name': '24/7',
                               'from_t': '00:00:00.000+00:00',
                               'to_t': '23:59:59.999+00:00',
                               'from_w': 'MONDAY',
                               'to_w': 'SUNDAY'})
        return print_xml_bimp(time_table)

    @staticmethod
    def _timetable_discoverer(calendar_path: Path, file: Path, sup, conf, mode, file_name=None) -> None:
        """Executes BIMP Simulations.
        """
        args = ['java', '-jar', str(calendar_path), str(file), sup, conf, str(mode)]
        if file_name:
            args.append(file_name)
        print_step(f'Timetable discovery, args = {args}')
        process = subprocess.run(args, check=True, stdout=subprocess.PIPE)
        found = False
        xml = [f'<qbp:timetables xmlns:qbp="{QBP_NAMESPACE_URI}">']
        for line in process.stdout.decode('utf-8').splitlines():
            if 'BIMP' in line:
                found = False if found else True
            if found and not 'BIMP' in line:
                xml.append(line.strip())
        xml.append('</qbp:timetables>')
        xml = etree.fromstring(''.join(xml).strip())
        ns = {'qbp': QBP_NAMESPACE_URI}
        # Fix timestamp format
        rules = (xml.find('qbp:timetable', namespaces=ns)
                 .find('qbp:rules', namespaces=ns)
                 .findall('qbp:rule', namespaces=ns))
        for rule in rules:
            rule.attrib['fromTime'] = (
                datetime.datetime.strptime(rule.attrib['fromTime'][:-6], "%H:%M:%S.%f").strftime("%H:%M:%S.%f+00:00"))
            rule.attrib['toTime'] = (
                datetime.datetime.strptime(rule.attrib['toTime'][:-6], "%H:%M:%S.%f").strftime("%H:%M:%S.%f+00:00"))
        if mode == 2:
            timetable = xml.find('qbp:timetable', namespaces=ns)
            timetable.attrib['default'] = "true"
        return xml
