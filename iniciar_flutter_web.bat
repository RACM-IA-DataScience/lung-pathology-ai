@echo off
title Lung Pathology AI - Flutter Web Server
echo ========================================================
echo  Iniciando Servidor Web Flutter en http://localhost:8080
echo  Autor: Roberto Alejandro Chagra Martinez
echo ========================================================
cd lung_pathology_ai
C:\Users\rober\.flutter\flutter\bin\flutter.bat run -d web-server --web-port=8080 --web-hostname=0.0.0.0
pause
