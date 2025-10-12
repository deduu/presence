#!/usr/bin/env python3
"""
System monitor to track CPU, memory, and thread usage.
Run this in a separate terminal while your face detection script is running.

Usage:
    python monitor.py <process_name>
    
Example:
    python monitor.py python  # Monitor all python processes
    python monitor.py face_detection.py  # Monitor specific script
"""

import psutil
import time
import sys
import os
from datetime import datetime


def clear_screen():
    """Clear terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')


def find_processes(name):
    """Find all processes matching the given name"""
    matching = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            proc_name = proc.info['name'].lower()
            cmdline = ' '.join(proc.info['cmdline'] or []).lower()

            if name.lower() in proc_name or name.lower() in cmdline:
                matching.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return matching


def format_bytes(bytes_val):
    """Format bytes into human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} TB"


def monitor_process(process, interval=1.0):
    """Monitor a single process"""
    print(f"\n📊 Monitoring: {process.name()} (PID: {process.pid})")
    print(f"Command: {' '.join(process.cmdline())}\n")

    try:
        while True:
            clear_screen()

            # Header
            print("="*70)
            print(
                f"Process Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("="*70)
            print(f"Name: {process.name()}")
            print(f"PID: {process.pid}")
            print(f"Status: {process.status()}")
            print("-"*70)

            # CPU Usage
            cpu_percent = process.cpu_percent(interval=0.1)
            num_threads = process.num_threads()
            print(
                f"CPU Usage:     {cpu_percent:6.2f}% ({num_threads} threads)")

            # Memory Usage
            mem_info = process.memory_info()
            mem_percent = process.memory_percent()
            print(
                f"Memory (RSS):  {format_bytes(mem_info.rss):>12} ({mem_percent:.2f}%)")
            print(f"Memory (VMS):  {format_bytes(mem_info.vms):>12}")

            # IO Stats (if available)
            try:
                io_counters = process.io_counters()
                print(
                    f"Disk Read:     {format_bytes(io_counters.read_bytes):>12}")
                print(
                    f"Disk Write:    {format_bytes(io_counters.write_bytes):>12}")
            except (AttributeError, psutil.AccessDenied):
                pass

            # System-wide CPU per core
            print("-"*70)
            print("System CPU Usage (per core):")
            per_core = psutil.cpu_percent(interval=0.1, percpu=True)
            for i, usage in enumerate(per_core):
                bar_length = int(usage / 2)  # Scale to 50 chars max
                bar = "█" * bar_length + "░" * (50 - bar_length)
                print(f"  Core {i:2d}: [{bar}] {usage:5.1f}%")

            # Overall system stats
            print("-"*70)
            cpu_freq = psutil.cpu_freq()
            print(
                f"CPU Frequency: {cpu_freq.current:.0f} MHz (Max: {cpu_freq.max:.0f} MHz)")
            print(f"System Load:   {psutil.cpu_percent(interval=0.1):.1f}%")

            mem = psutil.virtual_memory()
            print(
                f"System Memory: {format_bytes(mem.used)} / {format_bytes(mem.total)} ({mem.percent}%)")

            print("="*70)
            print("Press Ctrl+C to stop monitoring")

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n\n✅ Monitoring stopped.")
    except psutil.NoSuchProcess:
        print("\n\n❌ Process terminated.")


def main():
    if len(sys.argv) < 2:
        print("Usage: python monitor.py <process_name>")
        print("\nExample:")
        print("  python monitor.py python")
        print("  python monitor.py face_detection.py")
        sys.exit(1)

    process_name = sys.argv[1]

    print(f"🔍 Looking for processes matching: {process_name}")
    processes = find_processes(process_name)

    if not processes:
        print(f"❌ No processes found matching: {process_name}")
        sys.exit(1)

    if len(processes) == 1:
        monitor_process(processes[0])
    else:
        print(f"\n✅ Found {len(processes)} matching processes:")
        for i, proc in enumerate(processes, 1):
            try:
                cmdline = ' '.join(proc.cmdline()[:3])  # First 3 args
                print(f"  {i}. PID {proc.pid}: {cmdline}")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        choice = input(
            "\nSelect process number to monitor (or 'all' for summary): ")

        if choice.lower() == 'all':
            print("\n📊 Monitoring all matching processes...")
            try:
                while True:
                    clear_screen()
                    print("="*70)
                    print(
                        f"Multi-Process Monitor - {datetime.now().strftime('%H:%M:%S')}")
                    print("="*70)

                    for proc in processes:
                        try:
                            cpu = proc.cpu_percent(interval=0.1)
                            mem = proc.memory_info().rss / 1024 / 1024
                            threads = proc.num_threads()
                            print(
                                f"PID {proc.pid:6d}: CPU {cpu:5.1f}% | RAM {mem:7.1f}MB | Threads {threads:3d}")
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass

                    print("="*70)
                    print("Press Ctrl+C to stop")
                    time.sleep(1)

            except KeyboardInterrupt:
                print("\n\n✅ Monitoring stopped.")
        else:
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(processes):
                    monitor_process(processes[idx])
                else:
                    print("❌ Invalid selection")
            except ValueError:
                print("❌ Invalid input")


if __name__ == "__main__":
    main()
