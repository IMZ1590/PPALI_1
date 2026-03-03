from fastapi import FastAPI, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from typing import List
import os
import re
import numpy as np
import sys

# Add current directory to sys.path to ensure analyzer import works
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from analyzer import run_residue_pca

app = FastAPI(title="PALI 1")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Relative path to frontend
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_PATH = os.path.join(BASE_DIR, "frontend", "index.html")

@app.get("/logo")
async def get_logo():
    # Robust path resolution: Look for logo in the same directory as this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    logo_path = os.path.join(script_dir, "KBSI_Logo.png")
    if os.path.exists(logo_path):
        return FileResponse(logo_path)
    return FileResponse("KBSI_Logo.png") # Fallback

@app.get("/example-csv")
async def get_example_csv():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    example_path = os.path.join(script_dir, "Example_CSV.csv")
    if os.path.exists(example_path):
         return FileResponse(example_path)
    raise HTTPException(status_code=404, detail="Example CSV not found")

@app.get("/")
async def read_index():
    if not os.path.exists(FRONTEND_PATH):
        raise HTTPException(status_code=404, detail="Frontend file not found")
    return FileResponse(FRONTEND_PATH)

@app.post("/analyze_picked")
async def analyze_picked(
    intensities: List[str] = Form(...),
    calculate_csp: bool = Form(False)
):
    try:
        residue_data = []
        feature_names = None
        
        # If the frontend sends everything as ONE large string in a list, split it
        all_lines = []
        for item in intensities:
            all_lines.extend(item.replace('\r', '').split('\n'))
            
        # Parse lines
        valid_lines = [l.strip() for l in all_lines if l.strip()]
        if not valid_lines:
             raise HTTPException(status_code=400, detail="No data provided")
             
        first_line = valid_lines[0]
        parts = [p for p in re.split(r'[,\s]+', first_line) if p.strip()]
        
        # Check if first line is header (if 2nd col onwards are NOT floats)
        is_header = False
        if len(parts) >= 2:
            try:
                [float(v) for v in parts[1:]]
            except ValueError:
                is_header = True
                feature_names = parts[1:]
                
        # Parse data
        for s in valid_lines:
            if is_header and s == first_line:
                continue
                
            parts = [p for p in re.split(r'[,\s]+', s) if p.strip()]
            if len(parts) < 2: continue
            
            res_id_str_raw = parts[0]
            # Strip non-numeric characters to get pure residue number (e.g. "A10" -> "10")
            res_id_numeric = re.sub(r'\D', '', res_id_str_raw)
            
            if not res_id_numeric: 
                # If no number found, skip or keep raw? Let's skip to avoid NaN issues
                continue
                
            try:
                values = [float(v) for v in parts[1:]]
                # Use the processed numeric string as the ID
                residue_data.append([res_id_numeric] + values)
                
                # Fallback naming if no header
                if feature_names is None:
                     feature_names = [f"Feature_{i+1}" for i in range(len(values))]
            except ValueError:
                continue
        
        if not residue_data:
            raise HTTPException(status_code=400, detail="No valid residue data found")

        # Validate H/N for CSP
        h_idx = None
        n_idx = None
        
        if calculate_csp:
            if not is_header or len(feature_names) < 2:
                raise HTTPException(status_code=400, detail="To calculate CSP, the data must have headers and at least two feature columns (e.g., delta_H and delta_N).")
            
            # Find the indices of H and N among the first two columns (feature_names[0] and feature_names[1])
            col0 = feature_names[0].lower()
            col1 = feature_names[1].lower()
            
            # Use regex to strictly identify H and N (avoids matching 'n' in 'Intensity')
            def is_h_col(name): return bool(re.search(r'\b(h|1h|dh|delta_h)\b', name.replace('-','_'))) or name == 'h'
            def is_n_col(name): return bool(re.search(r'\b(n|15n|dn|delta_n)\b', name.replace('-','_'))) or name == 'n'
            
            # Also allow simple fallback if exactly matching:
            if not (is_h_col(col0) or is_n_col(col0)): 
                if 'h' in col0 and 'n' not in col0: col0_is_h = True
                else: col0_is_h = is_h_col(col0)
            else: col0_is_h = is_h_col(col0)
            
            if col0_is_h and is_n_col(col1):
                h_idx = 0
                n_idx = 1
            elif is_n_col(col0) and (is_h_col(col1) or 'h' in col1):
                h_idx = 1
                n_idx = 0
            else:
                raise HTTPException(status_code=400, detail=f"To calculate CSP, the first two feature columns must be H and N chemical shifts (in any order). Found: {feature_names[0]} and {feature_names[1]}")
        
        # Run PCA
        result = run_residue_pca(residue_data, feature_names=feature_names, calculate_csp=calculate_csp, h_idx=h_idx, n_idx=n_idx)
        return result
        
    except Exception as e:
        # import traceback
        # traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)