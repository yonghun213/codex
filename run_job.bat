@echo off
REM UTF-8 설정
chcp 65001 >nul

REM 현재 스크립트가 있는 폴더로 이동 (절대 경로 대신 권장)
cd /d "%~dp0"

REM 로그 폴더 생성
if not exist logs mkdir logs

REM [중요] 시스템 날짜 형식에 의존하지 않고 YYYYMMDD 구하기 (WMIC 사용)
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set datetime=%%I
set YYYYMMDD=%datetime:~0,8%

REM 로그 파일명 설정
set LOGFILE=logs\job_%YYYYMMDD%.log

echo [%DATE% %TIME%] 작업 시작 >> "%LOGFILE%"

REM 1단계 실행
echo ------------------------------------------------ >> "%LOGFILE%"
echo [1단계] 주문 데이터 가져오기 (load_orders1.py) >> "%LOGFILE%"
python load_orders1.py config.ini >> "%LOGFILE%" 2>&1

REM 에러 체크 및 2단계 실행
IF %ERRORLEVEL% EQU 0 (
    echo [2단계] 이메일 리포트 전송 (daily_sales_email1.py) >> "%LOGFILE%"
    python daily_sales_email1.py config.ini >> "%LOGFILE%" 2>&1
    echo [완료] 작업이 성공적으로 끝났습니다. >> "%LOGFILE%"
) ELSE (
    echo [오류] 1단계 실패. (%ERRORLEVEL%) 이메일을 발송하지 않습니다. >> "%LOGFILE%"
)
echo ================================================ >> "%LOGFILE%"

REM 아래 명령어를 추가하면 창이 바로 닫히지 않습니다.
pause
