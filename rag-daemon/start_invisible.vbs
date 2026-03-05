' Launches the Eureka host worker completely silently (no console window).
' Place a shortcut to this file in shell:startup to auto-run on login.
'
' To set up auto-start:
'   1. Press Win+R, type shell:startup, press Enter
'   2. Copy this file (or a shortcut to it) into that folder
'   3. On next login, the worker starts invisibly in the background
'   4. Logs are written to rag-daemon\logs\worker.log
Set WshShell = CreateObject("WScript.Shell")
WshShell.Run "C:\Users\madus\Desktop\Eureka\rag-daemon\start_eureka.bat", 0, False
