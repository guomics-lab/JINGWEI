#ifndef SourceDir
  #error SourceDir must point to the validated Windows distribution
#endif
#ifndef ReleaseDir
  #error ReleaseDir must point to the release output directory
#endif
#ifndef AppVersion
  #error AppVersion must match the application build
#endif

[Setup]
AppId={{91575868-B05E-4E1F-BD07-DA7D326C7329}
AppName=JINGWEI Benchmark
AppVersion={#AppVersion}
AppPublisher=Guomics Lab
AppPublisherURL=https://github.com/guomics-lab/JINGWEI
AppSupportURL=https://github.com/guomics-lab/JINGWEI/issues
DefaultDirName={localappdata}\Programs\JINGWEI Benchmark
DefaultGroupName=JINGWEI Benchmark
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
DisableProgramGroupPage=yes
DisableWelcomePage=no
WizardStyle=modern
SetupIconFile={#SourceDir}\jingwei.ico
UninstallDisplayIcon={app}\JINGWEI Benchmark.exe
LicenseFile=..\LICENSE
OutputDir={#ReleaseDir}
OutputBaseFilename=JINGWEI-Benchmark-{#AppVersion}-windows-x86_64-setup
VersionInfoVersion={#AppVersion}
VersionInfoDescription=JINGWEI Benchmark Setup
Compression=lzma2/normal
SolidCompression=yes
LZMAUseSeparateProcess=yes
LZMANumBlockThreads=4
DiskSpanning=yes
DiskSliceSize=1900000000
SlicesPerDisk=1
CloseApplications=yes
RestartApplications=no
Uninstallable=yes
UsePreviousAppDir=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "chinesesimp"; MessagesFile: "languages\ChineseSimplified.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\JINGWEI Benchmark"; Filename: "{app}\JINGWEI Benchmark.exe"; WorkingDir: "{userdocs}"; AppUserModelID: "JINGWEI.Benchmark.Desktop"
Name: "{autodesktop}\JINGWEI Benchmark"; Filename: "{app}\JINGWEI Benchmark.exe"; WorkingDir: "{userdocs}"; AppUserModelID: "JINGWEI.Benchmark.Desktop"; Tasks: desktopicon
Name: "{autoprograms}\JINGWEI Benchmark guide"; Filename: "https://github.com/guomics-lab/JINGWEI/blob/main/software/benchmark/README.md"

[Run]
Filename: "{app}\JINGWEI Benchmark.exe"; WorkingDir: "{userdocs}"; Description: "{cm:LaunchProgram,JINGWEI Benchmark}"; Flags: nowait postinstall skipifsilent
