flowchart LR
    subgraph P1["Phase 1: Implementation"]
        A1[Preprocessing] --> A2[EDA]
        A2 --> A3[Train/Test Split by Cell]
        A3 --> A4[Fit Nonlinear Model]
        A4 --> A5[Bootstrap + 95% CI]
        A5 --> A6[Evaluate: Residuals, N80]
    end
    
    subgraph P2["Phase 2: Modifications"]
        A6 --> B1{Diagnose Issues}
        B1 -->|Option 1| B2[Change Point Detection]
        B1 -->|Option 2| B3[PCA on T, C-rate, DoD]
        B1 -->|Option 3| B4[Add Electrolyte Factor]
        B2 & B3 & B4 --> B5[Refit & Re-evaluate]
    end
    
    subgraph P3["Phase 3: Justifications"]
        B5 --> C1[Compare Original vs Modified]
        C1 --> C2{Improvement?}
        C2 -->|Yes| C3[Recommend Modified Framework]
        C2 -->|No| C4[Identify Limitations]
        C3 & C4 --> C5[Final Report]
    end
