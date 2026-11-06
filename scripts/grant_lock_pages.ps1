<#
.SYNOPSIS
  Grants the "Lock pages in memory" right (SeLockMemoryPrivilege) to a user, so that pcuda can
  back large allocations with 2 MB pages (PCUDA_LARGE_PAGES=1).

  Works on Windows Home too (it has no secpol.msc): it calls LsaAddAccountRights directly.
  Run it from an elevated PowerShell, then sign out and back in: the right is added to the
  logon token at sign-in.

  Large-page memory is locked: it is never paged out while allocated. pcuda uses it only for
  allocations of 2 MB or more, and only when PCUDA_LARGE_PAGES=1.

.EXAMPLE
  .\scripts\grant_lock_pages.ps1              # the current user
  .\scripts\grant_lock_pages.ps1 -Revoke      # remove the right again
#>
[CmdletBinding()]
param(
    [string]$User = "$env:USERDOMAIN\$env:USERNAME",
    [switch]$Revoke
)
$ErrorActionPreference = 'Stop'

$admin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $admin) { throw 'Run this script from an elevated PowerShell ("Run as administrator").' }

Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
using System.Security.Principal;

public static class LsaRights {
    [StructLayout(LayoutKind.Sequential)]
    struct LSA_UNICODE_STRING { public ushort Length; public ushort MaximumLength; public IntPtr Buffer; }
    [StructLayout(LayoutKind.Sequential)]
    struct LSA_OBJECT_ATTRIBUTES {
        public int Length; public IntPtr RootDirectory; public IntPtr ObjectName;
        public int Attributes; public IntPtr SecurityDescriptor; public IntPtr SecurityQualityOfService;
    }
    [DllImport("advapi32.dll")] static extern uint LsaOpenPolicy(IntPtr system, ref LSA_OBJECT_ATTRIBUTES attrs, int access, out IntPtr handle);
    [DllImport("advapi32.dll")] static extern uint LsaAddAccountRights(IntPtr handle, byte[] sid, LSA_UNICODE_STRING[] rights, int count);
    [DllImport("advapi32.dll")] static extern uint LsaRemoveAccountRights(IntPtr handle, byte[] sid, bool all, LSA_UNICODE_STRING[] rights, int count);
    [DllImport("advapi32.dll")] static extern uint LsaClose(IntPtr handle);
    [DllImport("advapi32.dll")] static extern int LsaNtStatusToWinError(uint status);

    public static void Set(string account, string right, bool add) {
        var sid = (SecurityIdentifier)new NTAccount(account).Translate(typeof(SecurityIdentifier));
        var bytes = new byte[sid.BinaryLength];
        sid.GetBinaryForm(bytes, 0);
        var attrs = new LSA_OBJECT_ATTRIBUTES();
        IntPtr policy;
        uint st = LsaOpenPolicy(IntPtr.Zero, ref attrs, 0x00000800 | 0x00000010, out policy);  // LOOKUP_NAMES | CREATE_ACCOUNT
        if (st != 0) throw new System.ComponentModel.Win32Exception(LsaNtStatusToWinError(st));
        try {
            var s = new LSA_UNICODE_STRING();
            s.Buffer = Marshal.StringToHGlobalUni(right);
            s.Length = (ushort)(right.Length * 2);
            s.MaximumLength = (ushort)(right.Length * 2 + 2);
            try {
                st = add ? LsaAddAccountRights(policy, bytes, new[] { s }, 1)
                         : LsaRemoveAccountRights(policy, bytes, false, new[] { s }, 1);
                if (st != 0) throw new System.ComponentModel.Win32Exception(LsaNtStatusToWinError(st));
            } finally { Marshal.FreeHGlobal(s.Buffer); }
        } finally { LsaClose(policy); }
    }
}
'@

[LsaRights]::Set($User, 'SeLockMemoryPrivilege', -not $Revoke)
if ($Revoke) {
    Write-Host "Removed 'Lock pages in memory' from $User. Sign out and back in for it to take effect."
} else {
    Write-Host "Granted 'Lock pages in memory' to $User."
    Write-Host 'Sign out and back in, then run with PCUDA_LARGE_PAGES=1, for example:'
    Write-Host '  $env:PCUDA_LARGE_PAGES = "1"; .\build\windows-msvc-nvidia\pcuda_llm_bench.exe --paths simd,cuda'
}
