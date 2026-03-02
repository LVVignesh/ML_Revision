from ultralytics import YOLO

# 1. Load the base model
model = YOLO('yolov8n.pt')

# 2. Start the "Deep Learning" phase
# We increase 'epochs' to 50 so the AI has more time to find patterns
model.train(
    data=r'C:\Users\lvvig\ML_Revision_Lab\Phase3_Fast\data.yaml', 
    epochs=50,       # More practice rounds
    imgsz=640,       # High resolution for better detail
    name='strong_door_model' # Saves to a new folder so we don't mix them up
)

print("--- TRAINING FINISHED ---")
print("New brain is at: C:\\Users\\lvvig\\runs\\detect\\strong_door_model\\weights\\best.pt")