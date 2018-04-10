@ECHO OFF
SET MAIN=C:\Users\Asistente\Documents\Repositorio\PaperAnalysisVsSimulacion\Experimento\Implementacion\Simulacion\SiMo_v3
SET FILE=Production
REM SET FILE=ConsultaDataMining2016Apr2_1


CD %MAIN%
ECHO -- Mining Process Structure --
java -jar splitminer/splitminer.jar inputs/%FILE%.xes.gz inputs/%FILE%
CD %MAIN%
ECHO -- Process Alignment --
java -jar proconformance/ProConformance2.jar inputs/ %FILE%.xes.gz %FILE%.bpmn true
ECHO -- Mining Simulation Parameters --
python simo\simo.py -i inputs\%FILE%.xes.gz -b inputs\%FILE%.bpmn -o bimp\%FILE%.bpmn -r %MAIN%
ECHO -- Executing Simulations -- 
CD %MAIN%/bimp
FOR /l %%x IN (1, 1, 100) DO (
   ECHO Experiment #%%x
   java -jar qbp-simulator-engine.jar %FILE%.bpmn -csv %MAIN%/outputs/%%x.csv
)
CD %MAIN%
ECHO -- Data Comparison --
python analyzer\log_measurement.py
PAUSE