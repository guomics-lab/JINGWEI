#ifndef SourceDir
  #error SourceDir must point to the validated Windows distribution
#endif
#ifndef ReleaseDir
  #error ReleaseDir must point to the release output directory
#endif
#ifndef AppVersion
  #error AppVersion must match the application build
#endif
#ifndef InstallerDictionaryKB
  #define InstallerDictionaryKB 524288
#endif
#ifndef InstallerBlockThreads
  #define InstallerBlockThreads 2
#endif

[Setup]
AppId={{91575868-B05E-4E1F-BD07-DA7D326C7329}
AppName=jingwei
AppVersion={#AppVersion}
AppPublisher=Guomics Lab
AppPublisherURL=https://github.com/guomics-lab/JINGWEI
AppSupportURL=https://github.com/guomics-lab/JINGWEI/issues
DefaultDirName={localappdata}\Programs\jingwei
DefaultGroupName=jingwei
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
DisableProgramGroupPage=yes
DisableWelcomePage=no
DisableDirPage=no
WizardStyle=modern
SetupIconFile={#SourceDir}\jingwei.ico
UninstallDisplayIcon={app}\jingwei.exe
LicenseFile={#SourceDir}\LICENSE.txt
OutputDir={#ReleaseDir}
#ifdef SingleFileInstaller
OutputBaseFilename=jingwei-{#AppVersion}-windows-x86_64-offline
#else
OutputBaseFilename=jingwei-{#AppVersion}-windows-x86_64-setup
#endif
VersionInfoVersion={#AppVersion}
VersionInfoDescription=jingwei Setup
SolidCompression=yes
LZMAUseSeparateProcess=yes
#ifdef SingleFileInstaller
Compression=lzma2/ultra64
LZMADictionarySize={#InstallerDictionaryKB}
LZMANumBlockThreads={#InstallerBlockThreads}
DiskSpanning=no
#else
Compression=lzma2/normal
LZMANumBlockThreads=4
DiskSpanning=yes
DiskSliceSize=1900000000
SlicesPerDisk=1
#endif
CloseApplications=yes
RestartApplications=no
Uninstallable=not IsValidationInstall
UsePreviousAppDir=not IsValidationInstall

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "chinesesimp"; MessagesFile: "languages\ChineseSimplified.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[InstallDelete]
Type: files; Name: "{app}\JINGWEI Benchmark.exe"; Check: not IsValidationInstall
Type: files; Name: "{app}\jingwei-benchmark.exe"; Check: not IsValidationInstall
Type: files; Name: "{app}\LICENSE"; Check: not IsValidationInstall
Type: files; Name: "{autoprograms}\JINGWEI Benchmark.lnk"; Check: not IsValidationInstall
Type: files; Name: "{autoprograms}\JINGWEI Benchmark guide.url"; Check: not IsValidationInstall
Type: files; Name: "{autodesktop}\JINGWEI Benchmark.lnk"; Check: not IsValidationInstall

[Icons]
Name: "{autoprograms}\jingwei"; Filename: "{app}\jingwei.exe"; WorkingDir: "{userdocs}"; AppUserModelID: "JINGWEI.Benchmark.Desktop"; Check: not IsValidationInstall
Name: "{autodesktop}\jingwei"; Filename: "{app}\jingwei.exe"; WorkingDir: "{userdocs}"; AppUserModelID: "JINGWEI.Benchmark.Desktop"; Tasks: desktopicon; Check: not IsValidationInstall
Name: "{autoprograms}\jingwei guide"; Filename: "https://github.com/guomics-lab/JINGWEI/blob/main/software/benchmark/README.md"; Check: not IsValidationInstall

[Run]
Filename: "{app}\jingwei.exe"; WorkingDir: "{userdocs}"; Description: "{cm:LaunchProgram,jingwei}"; Flags: nowait postinstall skipifsilent; Check: not IsValidationInstall

[Code]
function IsValidationInstall: Boolean;
begin
  { Maintainer-only isolated extraction. No uninstall registry, shortcuts or
    automatic shutdown of the user's running app. The validator also passes
    /NOICONS, /NOCLOSEAPPLICATIONS and an explicit temporary /DIR. }
  Result := ExpandConstant('{param:JINGWEI-VALIDATION|0}') = '1';
end;
