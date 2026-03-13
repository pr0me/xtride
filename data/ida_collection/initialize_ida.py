#!/usr/bin/env python3
"""
Initialize IDA 9.0 for headless mode on macOS
Run this once before using the dataset generation scripts
"""

import os
import sys
import subprocess
import tempfile
import shutil

def initialize_ida(ida_path):
    """Initialize IDA settings for headless mode"""
    
    print("=== IDA 9.0 Initialization for Headless Mode ===\n")
    
    # Check if IDA exists
    if not os.path.exists(ida_path):
        print(f"❌ IDA not found at: {ida_path}")
        return False
    
    print(f"✓ Found IDA at: {ida_path}")
    
    # Setup IDA user directory
    home = os.path.expanduser("~")
    ida_user_dir = os.path.join(home, ".idapro")
    
    # Create IDA user directory if it doesn't exist
    if not os.path.exists(ida_user_dir):
        os.makedirs(ida_user_dir)
        print(f"✓ Created IDA user directory: {ida_user_dir}")
    else:
        print(f"✓ IDA user directory exists: {ida_user_dir}")
    
    # Create a minimal ida.reg file to prevent venv issues
    ida_reg_path = os.path.join(ida_user_dir, "ida.reg")
    if not os.path.exists(ida_reg_path):
        with open(ida_reg_path, 'w') as f:
            f.write("""
; IDA Pro Registry File
[IDA]
AUTOSAVE=1
ASCIISTRINGS=3
[IDAPYTHON]
IDAPYTHON_DISABLE_VENV=1
IDAPYTHON_USE_VENV=0
""")
        print(f"✓ Created minimal IDA registry: {ida_reg_path}")
    else:
        print(f"✓ IDA registry exists: {ida_reg_path}")
    
    # Create cfg directory for IDA configuration
    cfg_dir = os.path.join(ida_user_dir, "cfg")
    if not os.path.exists(cfg_dir):
        os.makedirs(cfg_dir)
        print(f"✓ Created IDA cfg directory: {cfg_dir}")
    
    # Test IDA with a simple binary
    print("\n=== Testing IDA Headless Mode ===\n")
    
    # Create a minimal test binary
    test_binary = "/tmp/test_ida_binary"
    with open(test_binary, 'wb') as f:
        # Minimal ELF64 header
        f.write(b'\x7fELF\x02\x01\x01' + b'\x00' * 9)  # ELF magic + 64-bit
        f.write(b'\x02\x00\x3e\x00')  # ET_EXEC, x86-64
        f.write(b'\x01\x00\x00\x00')  # version
        f.write(b'\x00' * 48)  # padding
    
    # Create test script
    test_script = """
import sys
print("IDA Python initialized successfully")
print(f"Python version: {sys.version}")
try:
    import idaapi
    print(f"IDA SDK version: {idaapi.IDA_SDK_VERSION}")
    idaapi.auto_wait()
    print("Auto-analysis complete")
    idaapi.qexit(0)
except Exception as e:
    print(f"Error: {e}")
    import idaapi
    idaapi.qexit(1)
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(test_script)
        script_path = f.name
    
    try:
        # Test IDA
        env = os.environ.copy()
        env['HOME'] = home
        env['IDAUSR'] = ida_user_dir
        # Clear Python environment variables
        env.pop('PYTHONPATH', None)
        env.pop('PYTHONHOME', None)
        env.pop('VIRTUAL_ENV', None)
        
        log_path = os.path.join(tempfile.gettempdir(), "ida_init_test.log")
        if os.path.exists(log_path):
            os.unlink(log_path)

        cmd = [ida_path, "-A", "-B", f"-L{log_path}", f"-S{script_path}", test_binary]
        print(f"Running: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            env=env
        )

        log_content = ""
        if os.path.exists(log_path):
            with open(log_path, 'r', errors='ignore') as f:
                log_content = f.read()
        
        if "IDA Python initialized successfully" in log_content:
            print("\n✅ IDA headless mode is working correctly!")
            print(f"Log content:\n---\n{log_content}\n---")
            return True
        else:
            print("\n❌ IDA headless mode test failed")
            print(f"Return code: {result.returncode}")
            print(f"IDA Log:\n---\n{log_content}\n---")
            if result.stdout:
                print(f"STDOUT:\n{result.stdout}")
            if result.stderr:
                print(f"STDERR:\n{result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("\n❌ IDA timed out - headless mode may not be working")
        return False
    except Exception as e:
        print(f"\n❌ Error testing IDA: {e}")
        return False
    finally:
        # Clean up
        if os.path.exists(script_path):
            os.unlink(script_path)
        if os.path.exists(test_binary):
            os.unlink(test_binary)
        # Clean up IDA database files
        for ext in ['.i64', '.id0', '.id1', '.id2', '.til', '.nam']:
            db_file = test_binary + ext
            if os.path.exists(db_file):
                os.unlink(db_file)

if __name__ == "__main__":
    ida_path = "/Applications/IDA Professional 9.0.app/Contents/MacOS/idat"
    if len(sys.argv) > 1:
        ida_path = sys.argv[1]
    
    if initialize_ida(ida_path):
        print("\n✅ IDA is ready for headless dataset generation!")
        print("\nYou can now run:")
        print(f"  python generate.py --ida '{ida_path}' -t 1 -v -b <binaries_dir> -o <output_dir>")
    else:
        print("\n⚠️  IDA initialization failed. Please check:")
        print("1. IDA is properly licensed/activated")
        print("2. IDA has been run at least once in GUI mode")
        print("3. System security settings allow IDA to run")
        print("4. Try running IDA GUI once to complete initial setup")
