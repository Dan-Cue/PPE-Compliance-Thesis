# sequential_checker.py
"""
Sequential PPE Verification System 
"""

import time
from enum import Enum

class PPECheckStage(Enum):
    """Stages of PPE checking in order"""
    HAIRCAP = 0
    MASK = 1
    LONG_SLEEVES = 2
    APRON = 3
    GLOVES = 4
    BOOTS = 5
    COMPLETE = 6


class SequentialPPEChecker:
    """
    Sequential PPE verification system.
    Checks one PPE item at a time in order.
    ALL items must be properly verified - NO SKIPPING!
    """
    
    def __init__(self, config):
        self.config = config
        
        # Build checking order based on requirements
        self.check_order = []
        
        if config.PPE_REQUIREMENTS.get("haircap", False):
            self.check_order.append(PPECheckStage.HAIRCAP)
        if config.PPE_REQUIREMENTS.get("mask", False):
            self.check_order.append(PPECheckStage.MASK)
        if config.PPE_REQUIREMENTS.get("long_sleeves", False):
            self.check_order.append(PPECheckStage.LONG_SLEEVES)
        if config.PPE_REQUIREMENTS.get("apron", False):
            self.check_order.append(PPECheckStage.APRON)
        if config.PPE_REQUIREMENTS.get("gloves", False):
            self.check_order.append(PPECheckStage.GLOVES)
        if config.PPE_REQUIREMENTS.get("boots", False):
            self.check_order.append(PPECheckStage.BOOTS)
        
        # Always add COMPLETE at the end
        self.check_order.append(PPECheckStage.COMPLETE)
        
        # Current stage index
        self.current_stage_index = 0
        self.current_stage = self.check_order[0] if self.check_order else PPECheckStage.COMPLETE
        
        # Track which items are ACTUALLY verified
        self.verified_items = {
            PPECheckStage.HAIRCAP: False,
            PPECheckStage.MASK: False,
            PPECheckStage.LONG_SLEEVES: False,
            PPECheckStage.APRON: False,
            PPECheckStage.GLOVES: False,
            PPECheckStage.BOOTS: False,
        }
        
        # Stability tracking
        self.stability_counter = 0
        self.required_stability = max(1, config.STABILITY_FRAMES)  # At least 1
        
        # Classifier tracking
        # classifier_checked: True after classifier has run and APPROVED the current stage attempt.  Reset to False on stage advance or failure. Use the should_run_classifier property to know when to trigger it.
        
        self.classifier_checked = False
        
        # Recheck tracking for verified items
        self.frames_since_verification = {stage: 0 for stage in PPECheckStage}
        self.recheck_interval = 30
        
        # Stage start time (for display only, not timeout)
        self.stage_start_time = time.time()
        
        print(f"\n{'='*60}")
        print(f"Sequential Checker Initialized")
        print(f"{'='*60}")
        print(f"Total required items: {len(self.check_order) - 1}") 
        print(f"\nChecking order:")
        for i, stage in enumerate(self.check_order[:-1], 1):
            stage_name = self._get_stage_name(stage)
            print(f"  {i}. {stage_name}")
        print(f"{'='*60}\n")
    
    def _get_stage_name(self, stage):
        """Get human-readable name for any stage."""
        names = {
            PPECheckStage.HAIRCAP: "Hair Cap",
            PPECheckStage.MASK: "Mask",
            PPECheckStage.LONG_SLEEVES: "Long Sleeves",
            PPECheckStage.APRON: "Apron",
            PPECheckStage.GLOVES: "Both Gloves",
            PPECheckStage.BOOTS: "Both Boots",
            PPECheckStage.COMPLETE: "Complete",
        }
        return names.get(stage, "Unknown")
    
    def get_current_stage_name(self):
        """Get name of current stage."""
        return self._get_stage_name(self.current_stage)
    
    def get_ppe_type_for_stage(self, stage):
        """Get PPE type string for a stage."""
        mapping = {
            PPECheckStage.HAIRCAP: "haircap",
            PPECheckStage.MASK: "mask",
            PPECheckStage.LONG_SLEEVES: "long_sleeves",
            PPECheckStage.APRON: "apron",
            PPECheckStage.GLOVES: "gloves",
            PPECheckStage.BOOTS: "boots",
        }
        return mapping.get(stage, None)
    
    def is_complete(self):
        """
        Check if truly complete.
        Must be at COMPLETE stage AND all required items verified.
        """
        # Must be at COMPLETE stage
        if self.current_stage != PPECheckStage.COMPLETE:
            return False
        
        # Verify all required items are actually verified
        for stage in self.check_order[:-1]:  # Exclude COMPLETE
            if not self.verified_items.get(stage, False):
                # Found unverified required item!
                print(f"⚠ WARNING: {self._get_stage_name(stage)} not verified but at COMPLETE!")
                return False
        
        return True
    
    def advance_to_next_stage(self):
        """
        Advance to next stage ONLY after current is verified.
        """
        # CRITICAL: Mark current stage as verified BEFORE advancing
        if self.current_stage != PPECheckStage.COMPLETE:
            self.verified_items[self.current_stage] = True
            print(f"✓✓✓ {self.get_current_stage_name()} VERIFIED! ✓✓✓")
        
        # Move to next stage
        if self.current_stage_index < len(self.check_order) - 1:
            self.current_stage_index += 1
            self.current_stage = self.check_order[self.current_stage_index]
            
            # Reset counters for new stage
            self.stability_counter = 0
            self.classifier_checked = False
            self.stage_start_time = time.time()
            
            if self.current_stage != PPECheckStage.COMPLETE:
                print(f"→ Now checking: {self.get_current_stage_name()}")
            else:
                print(f"\n{'='*60}")
                print(f"🎉 ALL PPE ITEMS VERIFIED! 🎉")
                print(f"{'='*60}\n")
    
    def process_verification(self, yolo_detected, keypoint_verified, classifier_verified=None):
        """
        Process verification for current stage.
        
        Args:
            yolo_detected: Bool - YOLO detected the item
            keypoint_verified: Bool - Keypoints verify positioning
            classifier_verified: Bool or None - Classifier confirms type
        
        Returns:
            Bool - True if advanced to next stage
        """
        # Already complete - do nothing
        if self.current_stage == PPECheckStage.COMPLETE:
            return False
        
        # STAGE 1: YOLO Detection
        if not yolo_detected:
            # Item not detected - reset and wait
            if self.stability_counter > 0:
                print(f"⚠ {self.get_current_stage_name()} lost YOLO detection - resetting")
            self.stability_counter = 0
            return False
        
        # STAGE 2: Keypoint Verification
        if not keypoint_verified:
            # Item detected but keypoints wrong - reset and wait
            if self.stability_counter > 0:
                print(f"⚠ {self.get_current_stage_name()} failed keypoint check - resetting")
            self.stability_counter = 0
            return False
        
        # Both YOLO and keypoints passed - increment stability
        self.stability_counter += 1
        
        # STAGE 3: Check if stable enough
        if self.stability_counter < self.required_stability:
            # Still building stability
            return False
        
        # STAGE 4: Classifier Verification (if enabled)
        if self.config.USE_CLASSIFIER:
            if not self.classifier_checked:
                # Waiting for classifier result
                if classifier_verified is None:
                    # Classifier hasn't run yet
                    return False
                
                # Classifier has returned a result
                self.classifier_checked = True
                
                if not classifier_verified:
                    # Classifier says WRONG item type
                    print(f"✗ Classifier REJECTED {self.get_current_stage_name()}")
                    self.stability_counter = 0
                    self.classifier_checked = False  # Allow retry
                    return False
                
                # Classifier verified! Fall through to advance
        
        # ALL CHECKS PASSED! Advance to next stage
        self.advance_to_next_stage()
        return True
    
    def recheck_verified_items(self, all_detections, verification_results):
        """
        Periodically recheck verified items to ensure still present.
        """
        for stage in self.check_order[:-1]:  # Exclude COMPLETE
            # Only check already verified items
            if not self.verified_items.get(stage, False):
                continue
            
            # Don't recheck current stage (actively checking)
            if stage == self.current_stage:
                continue
            
            # Increment counter
            self.frames_since_verification[stage] += 1
            
            # Check if time to recheck
            if self.frames_since_verification[stage] < self.recheck_interval:
                continue
            
            # Reset counter
            self.frames_since_verification[stage] = 0
            
            # Check if still present
            ppe_type = self.get_ppe_type_for_stage(stage)
            if not ppe_type:
                continue
            
            is_still_present = False
            
            if ppe_type in ["haircap", "mask", "long_sleeves", "apron"]:
                # Single items
                is_still_present = all_detections.get(ppe_type) is not None
            
            elif ppe_type == "gloves":
                # Both gloves must still be verified
                glove_result = verification_results.get("gloves", {})
                is_still_present = (glove_result.get("left") == "DETECTED_CORRECT" and
                                    glove_result.get("right") == "DETECTED_CORRECT")
            
            elif ppe_type == "boots":
                # Both boots must still be verified
                boot_result = verification_results.get("boots", {})
                is_still_present = (boot_result.get("left") == "DETECTED_CORRECT" and
                                    boot_result.get("right") == "DETECTED_CORRECT")
            
            # If lost, go back to that stage
            if not is_still_present:
                print(f"\n⚠⚠⚠ {self._get_stage_name(stage)} LOST! Going back... ⚠⚠⚠\n")
                self.verified_items[stage] = False
                
                # Find index of lost stage
                lost_stage_index = self.check_order.index(stage)
                
                # Go back to that stage
                self.current_stage_index = lost_stage_index
                self.current_stage = stage
                self.stability_counter = 0
                self.classifier_checked = False
                self.stage_start_time = time.time()
    
    def get_status_summary(self):
        """Get summary of verification status."""
        total = len(self.check_order) - 1  # Exclude COMPLETE
        verified = sum(1 for stage in self.check_order[:-1] 
                      if self.verified_items.get(stage, False))
        
        return {
            "total": total,
            "verified": verified,
            "current_stage": self.get_current_stage_name(),
            "is_complete": self.is_complete(),
            "progress_percent": (verified / total * 100) if total > 0 else 0
        }
    
    def get_elapsed_time(self):
        """Get time spent on current stage."""
        return time.time() - self.stage_start_time

    @property
    def should_run_classifier(self):
        """
        True only when the classifier should fire — ONCE per stage attempt.

        Stability must be fully built before this returns True, so the
        expensive TFLite inference is never triggered during the stability
        build-up phase. 
        """
        return (
            self.current_stage != PPECheckStage.COMPLETE
            and self.stability_counter >= self.required_stability
            and not self.classifier_checked
        )

    def reset(self):
        """Reset to beginning."""
        print("\n🔄 Resetting PPE checker to beginning...\n")
        
        self.current_stage_index = 0
        self.current_stage = self.check_order[0] if self.check_order else PPECheckStage.COMPLETE
        self.stability_counter = 0
        self.classifier_checked = False
        self.stage_start_time = time.time()
        
        # Clear all verifications
        for stage in self.verified_items:
            self.verified_items[stage] = False
        
        # Clear recheck counters
        for stage in self.frames_since_verification:
            self.frames_since_verification[stage] = 0
        
        print(f"Starting with: {self.get_current_stage_name()}\n")
