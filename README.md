# AI Study Mentor

Main project files are in:
- `Campus_X/LangGraph/projects/`

Project documentation:
- `Campus_X/LangGraph/projects/README.md`

Quick start:
```bash
cd Campus_X/LangGraph/projects
pip install -r requirements.txt
streamlit run app.py
```

## LangGraph Flow

```mermaid
flowchart LR
    A([START]) --> B[Intent Node]
    B -->|learn / quiz| C[Knowledge Tracker Node]
    B -->|awaiting answer| F[Evaluation Node]
    B -->|status| G[Memory Update Node]
    C -->|learn| D[Lesson Generator Node]
    C -->|quiz| E[Quiz Node]
    D --> G
    E --> G
    F --> G
    G --> H([END])
```
