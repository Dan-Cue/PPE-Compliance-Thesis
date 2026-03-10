"""
hardware.py — GPIO Hardware Controller
=======================================

OVERVIEW
--------
Controls a physical door-lock relay via Raspberry Pi 5 GPIO using the
gpiozero library. Designed for a PPE verification system where a door
is only unlocked once all required PPE items have been confirmed.
Falls back to a safe simulation mode automatically if GPIO hardware is
unavailable (e.g. during development on a non-Pi machine).


HARDWARE SETUP
--------------
  Pin:    GPIO 26  (Physical Pin 37)
  Library: gpiozero LED(26, active_high=False)

Wiring (active-low relay):
  Pin 37  (GPIO 26)  ──→  Relay IN
  Pin 39  (GND)      ──→  Relay GND


SIGNAL LOGIC
------------
The relay is wired active-low, meaning the pin voltage level is
the inverse of the relay state:

  GPIO HIGH  →  Relay OFF  →  Door LOCKED    (startup / idle / reset)
  GPIO LOW   →  Relay ON   →  Door UNLOCKED  (all PPE verified)

gpiozero's active_high=False flag handles this inversion transparently:
  .off()  →  pin HIGH  →  Relay OFF  →  Door LOCKED   (safe default)
  .on()   →  pin LOW   →  Relay ON   →  Door UNLOCKED


INSTALLATION
------------
  pip install gpiozero

If gpiozero is not installed the module prints a warning and all
HardwareController instances automatically enter simulation mode.


CLASS: HardwareController
--------------------------
The main class. Manages relay state and GPIO lifecycle.

  __init__(pin, simulate)
    Initialises the GPIO pin and ensures the relay starts in the
    locked (HIGH) state. If GPIO setup fails for any reason, the
    instance silently degrades to simulation mode.

  activate_relay()
    Drives GPIO LOW → Relay ON → Door UNLOCKED.
    Called by the main application when all PPE items are verified.
    No-op if the relay is already active.

  deactivate_relay()
    Drives GPIO HIGH → Relay OFF → Door LOCKED.
    Called on system reset or program exit.
    No-op if the relay is already inactive.

  set_high() / set_low()
    Aliases for activate_relay() / deactivate_relay() so that existing
    main.py and test.py callers require no changes.

  is_active  (property)
    Returns True if the relay is currently in the ON (unlocked) state.

  cleanup()
    Forces GPIO HIGH (relay OFF), closes the gpiozero device, and
    releases the pin. Should be called on program exit.

  __enter__ / __exit__
    Implements the context manager protocol so the controller can be
    used in a with statement; cleanup() is called automatically on exit.


SIMULATION MODE
---------------
Activated automatically when:
  • gpiozero is not installed, or
  • GPIO initialisation raises an exception, or
  • simulate=True is passed to __init__.

In simulation mode all relay state changes are printed to stdout
with a [SIM] prefix and no hardware is touched. This allows the
full application to run and be tested on any machine.


SAFETY DESIGN
-------------
The system is fail-safe by default:
  • The relay starts locked (HIGH) on initialisation.
  • cleanup() always re-locks before releasing the pin.
  • Any GPIO exception during activate or deactivate is caught and
    logged without crashing the application.
  • Simulation mode ensures the rest of the system keeps running even
    if the GPIO subsystem is entirely unavailable.


DEPENDENCIES
------------
  gpiozero  — GPIO abstraction library (Raspberry Pi)
"""

GPIO_AVAILABLE = False

try:
    from gpiozero import LED
    GPIO_AVAILABLE = True
    print("✓  gpiozero imported successfully")
except ImportError:
    print("⚠  gpiozero not installed — running in SIMULATION mode")
    print("   Install with: pip install gpiozero")


GPIO_PIN = 26


class HardwareController:
    def __init__(self, pin=GPIO_PIN, simulate=False):
        self.pin      = pin
        self.simulate = simulate or not GPIO_AVAILABLE
        self._active  = False
        self._led     = None

        if self.simulate:
            print(f"ℹ  HardwareController: SIMULATION mode (GPIO {self.pin})")
        else:
            try:
                self._led = LED(self.pin, active_high=False)
                self._led.off()
                print(f"✓  GPIO {self.pin} ready → HIGH (Relay OFF — Door LOCKED)")
            except Exception as e:
                print(f"✗  GPIO init failed: {e}")
                print("   → Falling back to SIMULATION mode")
                self.simulate = True
                self._led = None

    def activate_relay(self):
        """GPIO LOW → Relay ON → Door UNLOCKED. Called when all PPE verified."""
        if self._active:
            return
        self._active = True
        if self.simulate:
            print(f"[SIM] GPIO {self.pin} → LOW | Relay ON → Door UNLOCKED ✓")
        else:
            try:
                self._led.on()
                print(f"✓  GPIO {self.pin} → LOW | Relay ON → Door UNLOCKED")
            except Exception as e:
                print(f"✗  GPIO activate failed: {e}")

    def deactivate_relay(self):
        """GPIO HIGH → Relay OFF → Door LOCKED. Called on system reset."""
        if not self._active:
            return
        self._active = False
        if self.simulate:
            print(f"[SIM] GPIO {self.pin} → HIGH | Relay OFF → Door LOCKED ↺")
        else:
            try:
                self._led.off()
                print(f"✓  GPIO {self.pin} → HIGH | Relay OFF → Door LOCKED")
            except Exception as e:
                print(f"✗  GPIO deactivate failed: {e}")

    # Aliases so main.py and test.py need zero changes
    def set_high(self):
        self.activate_relay()

    def set_low(self):
        self.deactivate_relay()

    @property
    def is_active(self):
        return self._active

    def cleanup(self):
        """Force GPIO HIGH (relay OFF) and release pin on program exit."""
        if not self.simulate and self._led is not None:
            try:
                self._led.off()
                self._led.close()
                print(f"✓  GPIO {self.pin} released (Relay OFF — Door LOCKED)")
            except Exception as e:
                print(f"⚠  GPIO cleanup warning: {e}")
        self._active = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False
