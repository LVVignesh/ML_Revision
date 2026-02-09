import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# =========================================================
# STEP 1: DATA PREP
# =========================================================
df = pd.read_csv("online_shoppers_intention.csv")
df["Revenue"] = df["Revenue"].astype(int)
df = pd.get_dummies(df, drop_first=True)

X = df.drop("Revenue", axis=1).values
y = df["Revenue"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

input_raw_dim = X_train.shape[1] 
print(f"Initial Input Features: {input_raw_dim}")

# =========================================================
# STEP 2: MULTI-HEAD ATTENTION (THE "EYES")
# =========================================================
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embed dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)

        self.out = nn.Linear(embed_dim, embed_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B = x.shape[0]
        # Reshape to (Batch, Heads, Head_Dim)
        Q = self.q(x).view(B, self.num_heads, self.head_dim)
        K = self.k(x).view(B, self.num_heads, self.head_dim)
        V = self.v(x).view(B, self.num_heads, self.head_dim)

        # Dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        weights = self.softmax(scores)

        attention = torch.matmul(weights, V)
        attention = attention.reshape(B, -1) # Flatten heads back together

        return self.out(attention)

# =========================================================
# STEP 3: TRANSFORMER BLOCK (ATTENTION + FFN + RESIDUALS)
# =========================================================
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 1. Attention + Residual + Norm
        attn_out = self.attention(x)
        x = self.norm1(x + self.dropout(attn_out)) # "Add & Norm"
        
        # 2. FFN + Residual + Norm
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))  # "Add & Norm"
        return x

# =========================================================
# STEP 4: FINAL CLASSIFIER
# =========================================================
class TransformerClassifier(nn.Module):
    def __init__(self, input_raw_dim, embed_dim=32, num_heads=4):
        super().__init__()
        # Projection: Force any input size into a size divisible by num_heads
        self.projection = nn.Linear(input_raw_dim, embed_dim)
        
        self.transformer = TransformerBlock(embed_dim, num_heads)
        
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.projection(x)
        x = self.transformer(x)
        return self.classifier(x)

# =========================================================
# STEP 5: TRAINING LOOP
# =========================================================
model = TransformerClassifier(input_raw_dim, embed_dim=32, num_heads=4)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 50
print("\nStarting Training...")
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch:02d} | Loss: {loss.item():.4f}")

# =========================================================
# STEP 6: EVALUATION
# =========================================================
model.eval()
with torch.no_grad():
    preds = model(X_test)
    preds_binary = (preds > 0.5).int()
    acc = accuracy_score(y_test, preds_binary)

print(f"\nTRANSFORMER TEST ACCURACY: {acc:.4f}")