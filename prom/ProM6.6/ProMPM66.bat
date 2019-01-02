@setlocal enableextensions
@cd /d "%~dp0"
jre7\bin\java -da -Xmx768M -XX:MaxPermSize=256m -classpath "ProM66.jar;ProM66_lib/*" -Djava.library.path=.//ProM66_lib org.processmining.contexts.uitopia.packagemanager.PMFrame
