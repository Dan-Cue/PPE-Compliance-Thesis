This is the code for the system of the prototype of DEVELOPMENT OF REAL-TIME PERSONAL PROTECTIVE EQUIPMENT (PPE) 
COMPLIANCE SYSTEM USING DEEP LEARNING-BASED POSE VERIFICATION

This system uses a pipeline that handles the verification of PPE in relevance to the body part.
The PPEs are:
White Hair Cap, White Cloth Mask, White Long Sleeved Shirt, White Apron, White Safety Gloves, and White Safety Boots. A total of 6 classes.
The system will verify each PPE sequentially starting from the Hair Cap down to the Boots. The verification cannot proceed to the next PPE unless the current one is Verified. This prevents faulty checks
and improves system reliability. The system is equipped with both Visual and Audio Feedback.
The system uses two models: YOLO model trained for these 6 classes and a pre-trained MediaPipe Pose Estimator Model. 

Here is the process of the pipeline:

MediaPipe checks every frame if there is a human present by tracking Human Body Keypoints (33 anatomical keypoints).
YOLO checks the frame if the item (PPE stage) is present.
Once YOLO detects, the system then calculates if the bounding box created by the YOLO model's detection overlaps with the keypoints tracked by MediaPipe.
This check happens 5 frames (5 times) and all of those frames must have correct (true) results and flags it as VERIFIED.
Then the system proceeds to the next stage (next PPE class) and this process repeats until all PPE are verified. 
Once every PPE was verified, the system will then activate a relay that is connected to the entrance lock. This allows entry control via PPE-Compliance with a pose verification layer.

Scenario Examples:

Scenario 1: All PPE Complete

MediaPipe (MP) detects Human (H1) -> MP locks on that human (closest and most relevant) and estimataes pose -> YOLO checks first PPE (Haircap) on the frame -> PPE detected -> System checks and verifies PPE for 5 frame -> PPE verified -> proceeds to the next PPE (Mask) -> Loop until all PPE are verified


Scenario 2: All but 1 PPE is correct

Same on Scenario 1 but left hand glove is missing. The verification will get stuck on the Gloves stage and loops until both left and right gloves are detected or when no human pose is present on the frame which then resets the verification to the beginning stage (haircap) after 5 seconds.


Scenario 3: No PPE

System will track human pose if present but loops on the first stage (haircap) and will never proceed to the next stages.
