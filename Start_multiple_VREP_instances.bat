@echo off
%SystemRoot%\System32\choice.exe /C YN /N /M "Do you want to start the instances headless [Y/N]?"
if not errorlevel 1 goto headless
if errorlevel 2 goto head

:headless
SET /p "n=Number of instances to open: "
set /A num=%n%
set /A end=%num%+19999

FOR /L %%G IN (20000,1,%end%) DO (
	start /d "C:\Program Files\V-REP3\V-REP_PRO_EDU" vrep.exe -h "C:\Program Files\V-REP3\V-REP_PRO_EDU\scenes\arena_push_easy.ttt" -gREMOTEAPISERVERSERVICE_%%G_FALSE_TRUE
)
goto :EOF
:head
SET /p "n=Number of instances to open: "
set /A num=%n%
set /A end=%num%+19999

FOR /L %%G IN (20000,1,%end%) DO (
	start /d "C:\Program Files\V-REP3\V-REP_PRO_EDU" vrep.exe "C:\Program Files\V-REP3\V-REP_PRO_EDU\scenes\arena_push_easy.ttt" -gREMOTEAPISERVERSERVICE_%%G_FALSE_TRUE
)
goto :EOF
