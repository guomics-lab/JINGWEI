#ifndef AppVersion
  #error AppVersion is required
#endif

[Setup]
AppId=JINGWEI.Benchmark.Download
AppName=JINGWEI Benchmark
AppVersion={#AppVersion}
AppPublisher=Guomics Lab
AppPublisherURL=https://github.com/guomics-lab/JINGWEI
DefaultDirName={localappdata}\Programs\JINGWEI Benchmark
CreateAppDir=no
Uninstallable=no
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
WizardStyle=modern
DisableWelcomePage=yes
DisableDirPage=yes
DisableProgramGroupPage=yes
DisableReadyPage=yes
DisableFinishedPage=yes
DisableStartupPrompt=yes
ShowLanguageDialog=yes
SetupIconFile={#AppIcon}
OutputDir={#ReleaseDir}
OutputBaseFilename=JINGWEI-Benchmark-Windows-Setup
VersionInfoVersion={#AppVersion}
VersionInfoDescription=JINGWEI Benchmark Windows Installer

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "chinesesimp"; MessagesFile: "languages\ChineseSimplified.isl"

[CustomMessages]
english.DownloadStatus=Downloading JINGWEI Benchmark...
chinesesimp.DownloadStatus=正在下载 JINGWEI Benchmark…
english.InstallStatus=Starting JINGWEI Benchmark Setup...
chinesesimp.InstallStatus=正在启动 JINGWEI Benchmark 安装程序…

[Files]
#include PayloadInclude

[Code]
var
  PayloadExitCode: Integer;

function DownloadCache(Param: String): String;
begin
  Result := ExpandConstant('{param:CACHE|}');
  if Result = '' then
    Result := ExpandConstant('{localappdata}\JINGWEI\Benchmark\Downloads\{#AppVersion}');
end;

function NeedDownload(Name, SHA256: String): Boolean;
var
  FileName: String;
begin
  Result := True;
  FileName := AddBackslash(DownloadCache('')) + Name;
  if FileExists(FileName) then begin
    try
      Result := CompareText(GetSHA256OfFile(FileName), SHA256) <> 0;
    except
      Result := True;
    end;
  end;
end;

procedure CheckPayload(Name, SHA256: String);
begin
  if NeedDownload(Name, SHA256) then
    RaiseException('JINGWEI download verification failed: ' + Name);
end;

#include VerifyInclude

procedure InitializeWizard;
begin
  PayloadExitCode := 1;
  WizardForm.StatusLabel.Caption := CustomMessage('DownloadStatus');
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  Arguments, Destination, PayloadLog, Payload: String;
  Code: Integer;
begin
  if CurStep = ssPostInstall then begin
    VerifyPayload;
    Payload := AddBackslash(DownloadCache('')) + '{#PayloadExe}';
    Arguments := '/SP- /NORESTART /LANG=' + ActiveLanguage;
    if WizardSilent then
      Arguments := Arguments + ' /VERYSILENT /SUPPRESSMSGBOXES';
    Destination := ExpandConstant('{param:DIR|}');
    if Destination <> '' then
      Arguments := Arguments + ' /DIR="' + Destination + '"';
    PayloadLog := ExpandConstant('{param:PAYLOADLOG|}');
    if PayloadLog <> '' then
      Arguments := Arguments + ' /LOG="' + PayloadLog + '"';
    WizardForm.StatusLabel.Caption := CustomMessage('InstallStatus');
    if not WizardSilent then
      WizardForm.Hide;
    if not Exec(Payload, Arguments, DownloadCache(''), SW_SHOWNORMAL, ewWaitUntilTerminated, Code) then begin
      PayloadExitCode := 1;
      RaiseException(SysErrorMessage(Code));
    end;
    PayloadExitCode := Code;
    if Code = 0 then
      DeletePayload;
  end;
end;

function GetCustomSetupExitCode: Integer;
begin
  Result := PayloadExitCode;
end;
