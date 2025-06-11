try:
    from call import emergency_call_demo
    SYSTEM_AVAILABLE = True
    print("[INFO] Emergency call system connected")
except ImportError:
    SYSTEM_AVAILABLE = False
    print("[WARNING] Emergency call system connection failed")

# Track which IDs have already triggered a call
ecall_sent_ids = set()


def try_emergency_call(person_id):
    if not SYSTEM_AVAILABLE or person_id in ecall_sent_ids:
        return False
    ecall_sent_ids.add(person_id)
    try:
        emergency_call_demo()
        print(f"📞 [CALL SENT] ID {person_id} Emergency call sent successfully")
        return True
    except Exception as e:
        print(f"❌ [CALL ERROR] {e}")
        return False


def reset_calls():
    ecall_sent_ids.clear()
