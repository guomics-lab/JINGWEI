#ifndef AppVersion
  #error AppVersion is required
#endif
#ifndef BootstrapFileName
  #define BootstrapFileName "jingwei-Windows-Setup"
#endif

[Setup]
AppId=JINGWEI.Benchmark.Download
AppName=jingwei
AppVersion={#AppVersion}
AppPublisher=Guomics Lab
AppPublisherURL=https://github.com/guomics-lab/JINGWEI
DefaultDirName={code:DefaultInstallDir}
CreateAppDir=yes
Uninstallable=no
UsePreviousAppDir=no
UsePreviousTasks=no
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
WizardStyle=modern
DisableWelcomePage=yes
DisableDirPage=no
DisableProgramGroupPage=yes
DisableReadyPage=no
DisableFinishedPage=no
DisableStartupPrompt=yes
ShowLanguageDialog=yes
SetupIconFile={#AppIcon}
LicenseFile={#AppLicense}
ExtraDiskSpaceRequired={#InstalledSize}
CloseApplications=no
OutputDir={#ReleaseDir}
OutputBaseFilename={#BootstrapFileName}
VersionInfoVersion={#AppVersion}
VersionInfoDescription=jingwei Windows Installer

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "chinesesimp"; MessagesFile: "languages\ChineseSimplified.isl"

[CustomMessages]
english.DownloadStatus=Downloading jingwei...
chinesesimp.DownloadStatus=正在下载 jingwei…
english.InstallStatus=Installing jingwei...
chinesesimp.InstallStatus=正在安装 jingwei…
english.InstallFailed=jingwei installation failed (code %1). Please run Setup again to retry.
chinesesimp.InstallFailed=jingwei 安装失败（代码 %1）。请重新运行安装程序重试。

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
#include PayloadInclude

[Run]
Filename: "{app}\jingwei.exe"; WorkingDir: "{userdocs}"; Description: "{cm:LaunchProgram,jingwei}"; Flags: nowait postinstall skipifsilent; Check: CanLaunch

[Code]
var
  PayloadExitCode: Integer;

function IsValidationInstall: Boolean;
begin
  Result := ExpandConstant('{param:JINGWEI-VALIDATION|0}') = '1';
end;

function DefaultInstallDir(Param: String): String;
var
  PreviousDir: String;
begin
  Result := ExpandConstant('{localappdata}\Programs\jingwei');
  { The download wrapper never registers an application. Read the real
    application's location so upgrades start at the existing directory,
    while keeping the directory page visible and editable. }
  if not IsValidationInstall then begin
    if not RegQueryStringValue(HKCU64,
      'Software\Microsoft\Windows\CurrentVersion\Uninstall\{91575868-B05E-4E1F-BD07-DA7D326C7329}_is1',
      'Inno Setup: App Path', PreviousDir) then
      RegQueryStringValue(HKCU32,
      'Software\Microsoft\Windows\CurrentVersion\Uninstall\{91575868-B05E-4E1F-BD07-DA7D326C7329}_is1',
      'Inno Setup: App Path', PreviousDir);
    if PreviousDir <> '' then
      Result := PreviousDir;
  end;
end;

function CanLaunch: Boolean;
begin
  Result := (PayloadExitCode = 0) and not IsValidationInstall;
end;

function DownloadCache(Param: String): String;
begin
  Result := ExpandConstant('{param:CACHE|}');
  if Result = '' then
    { The optional online installer stages its payload on the selected
      installation volume, never in a multi-GB C-drive user cache. }
    Result := AddBackslash(WizardDirValue) + '.jingwei-installer-cache\{#AppVersion}';
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
    { This wizard owns all user interaction. The offline payload installs
      exactly once without showing another wizard or launching the GUI. }
    Arguments := '/SP- /NORESTART /VERYSILENT /SUPPRESSMSGBOXES /LANG=' + ActiveLanguage;
    Arguments := Arguments + ' /TASKS="' + WizardSelectedTasks(False) + '"';
    if WizardNoIcons then
      Arguments := Arguments + ' /NOICONS';
    if IsValidationInstall then
      Arguments := Arguments + ' /JINGWEI-VALIDATION=1 /NOICONS /NOCLOSEAPPLICATIONS';
    Destination := WizardDirValue;
    Arguments := Arguments + ' /DIR="' + Destination + '"';
    PayloadLog := ExpandConstant('{param:PAYLOADLOG|}');
    if PayloadLog <> '' then
      Arguments := Arguments + ' /LOG="' + PayloadLog + '"';
    WizardForm.StatusLabel.Caption := CustomMessage('InstallStatus');
    WizardForm.FilenameLabel.Caption := Destination;
    WizardForm.ProgressGauge.Style := npbstMarquee;
    WizardForm.CancelButton.Enabled := False;
    try
      Log('Installing payload once into: ' + Destination);
      if not Exec(Payload, Arguments, DownloadCache(''), SW_HIDE, ewWaitUntilTerminated, Code) then begin
        PayloadExitCode := 1;
        RaiseException(SysErrorMessage(Code));
      end;
      PayloadExitCode := Code;
      if Code <> 0 then
        RaiseException(FmtMessage(CustomMessage('InstallFailed'), [IntToStr(Code)]));
      DeletePayload;
      RemoveDir(DownloadCache(''));
      if ExpandConstant('{param:CACHE|}') = '' then
        RemoveDir(AddBackslash(WizardDirValue) + '.jingwei-installer-cache');
    finally
      WizardForm.ProgressGauge.Style := npbstNormal;
      WizardForm.CancelButton.Enabled := True;
    end;
  end;
end;

function GetCustomSetupExitCode: Integer;
begin
  Result := PayloadExitCode;
end;
