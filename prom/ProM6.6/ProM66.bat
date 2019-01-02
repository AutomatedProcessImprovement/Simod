@setlocal enableextensions
@cd /d "%~dp0"
jre7\bin\java -da -Xmx4G -XX:MaxPermSize=256m -classpath ProM66.jar -Djava.util.Arrays.useLegacyMergeSort=true -DsuppressSwingDropSupport=false org.processmining.contexts.uitopia.UI
