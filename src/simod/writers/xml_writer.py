import uuid

from lxml import etree
from lxml.builder import ElementMaker  # lxml only !


# --------------- General methods ----------------
from simod.configuration import QBP_NAMESPACE_URI


def create_file(output_file, element):
    #    file_exist = os.path.exists(output_file)
    with open(output_file, 'wb') as f:
        f.write(element)
    f.close()


# -------------- kernel --------------
def print_parameters(bpmn_input, output_file, parameters):
    my_doc = xml_template(parameters.get('arrival_rate'),
                          parameters.get('resource_pool'),
                          parameters.get('elements_data'),
                          parameters.get('sequences'),
                          parameters.get('instances'),
                          parameters.get('start_time'))
    # insert timetable
    if parameters.get('time_table') is not None:
        ns = {'qbp': QBP_NAMESPACE_URI}
        childs = parameters['time_table'].findall('qbp:timetable', namespaces=ns)
        node = my_doc.find('qbp:timetables', namespaces=ns)
        for i, child in enumerate(childs):
            node.insert((i + 1), child)

    # Append parameters to the bpmn model
    root = append_parameters(bpmn_input, my_doc)
    create_file(output_file, etree.tostring(root, pretty_print=True))


def xml_template(arrival_rate=None, resource_pool=None, elements_data=None, sequences=None, instances=None,
                 start_time=None):
    E = ElementMaker(namespace=QBP_NAMESPACE_URI,
                     nsmap={'qbp': QBP_NAMESPACE_URI})
    PROCESSSIMULATIONINFO = E.processSimulationInfo
    ARRIVALRATEDISTRIBUTION = E.arrivalRateDistribution
    TIMEUNIT = E.timeUnit
    TIMETABLES = E.timetables
    RESOURCES = E.resources
    RESOURCE = E.resource
    ELEMENTS = E.elements
    ELEMENT = E.element
    DURATION = E.durationDistribution
    RESOURCESIDS = E.resourceIds
    RESOURCESID = E.resourceId
    SEQUENCEFLOWS = E.sequenceFlows
    SEQUENCEFLOW = E.sequenceFlow

    rootid = "qbp_" + str(uuid.uuid4())

    arrival_doc = None
    if arrival_rate:
        arrival_doc = ARRIVALRATEDISTRIBUTION(
            TIMEUNIT("seconds"),
            type=arrival_rate['dname'],
            mean=str(arrival_rate['dparams']['mean']),
            arg1=str(arrival_rate['dparams']['arg1']),
            arg2=str(arrival_rate['dparams']['arg2']))

    resources_doc = None
    if resource_pool:
        resources_doc = RESOURCES(
            *[
                RESOURCE(
                    id=res['id'],
                    name=res['name'],
                    totalAmount=res['total_amount'],
                    costPerHour=res['costxhour'],
                    timetableId=res['timetable_id']) for res in resource_pool
            ]
        )

    elements_doc = None
    if elements_data:
        elements_doc = ELEMENTS(
            *[
                ELEMENT(
                    DURATION(
                        TIMEUNIT("seconds"),
                        type=e['type'],
                        mean=e['mean'],
                        arg1=e['arg1'],
                        arg2=e['arg2']
                    ),
                    RESOURCESIDS(
                        RESOURCESID(str(e['resource']))
                    ),
                    id=e['id'], elementId=e['elementid']
                ) for e in elements_data
            ]
        )

    sequences_doc = None
    if sequences:
        sequences_doc = SEQUENCEFLOWS(
            *[
                SEQUENCEFLOW(
                    elementId=seq['elementid'],
                    executionProbability=str(seq['prob'])) for seq in sequences
            ]
        )

    docs = list(filter(lambda doc: doc is not None, [arrival_doc, resources_doc, elements_doc, sequences_doc]))
    my_doc = PROCESSSIMULATIONINFO(
        TIMETABLES(),
        *docs,
        id=rootid,
        processInstances=str(instances) if instances else "",
        startDateTime=start_time if start_time else "",
        currency="EUR"
    )
    return my_doc


def append_parameters(bpmn_input, my_doc):
    node = etree.fromstring(etree.tostring(my_doc, pretty_print=True))
    tree = etree.parse(bpmn_input)
    root = tree.getroot()
    froot = etree.fromstring(etree.tostring(root, pretty_print=True))
    froot.append(node)
    return froot
