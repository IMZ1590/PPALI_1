from fastapi import FastAPI, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import urllib.request
import urllib.parse
from datetime import datetime
from fastapi.responses import FileResponse
from typing import List
import os
import re
import numpy as np
import sys

# Add current directory to sys.path to ensure analyzer import works
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from analyzer import run_residue_pca

def run_blast_and_rename(file_path: str, sequence: str):
    date_str = datetime.now().strftime("%Y%m%d")
    dir_name = os.path.dirname(file_path)
    base_name = os.path.basename(file_path)
    
    # Extract unique ID from original filename (data_{timestamp}_{uuid}.csv)
    parts = base_name.split('_')
    if len(parts) >= 3:
        unique_num = parts[2].split('.')[0]
    else:
        import uuid
        unique_num = str(uuid.uuid4())[:8]
    
    def rename_file(new_name):
        new_path = os.path.join(dir_name, new_name)
        if os.path.exists(file_path):
            os.rename(file_path, new_path)
            
    if not sequence or len(sequence) < 10:
        rename_file(f"{date_str}_{unique_num}_unknown.csv")
        return

    try:
        params = {
            'CMD': 'Put',
            'PROGRAM': 'blastp',
            'DATABASE': 'nr',
            'QUERY': sequence,
            'EXPECT': 100.0
        }
        data = urllib.parse.urlencode(params).encode('utf-8')
        req = urllib.request.Request("https://blast.ncbi.nlm.nih.gov/blast/Blast.cgi", data=data)
        
        with urllib.request.urlopen(req) as response:
            html = response.read().decode('utf-8')
            match = re.search(r'RID = (.*)', html)
            if not match:
                rename_file(f"{date_str}_{unique_num}_unknown.csv")
                return
            rid = match.group(1).strip()
            
        import time
        for _ in range(30):
            time.sleep(10)
            get_params = {
                'CMD': 'Get',
                'FORMAT_TYPE': 'XML',
                'RID': rid
            }
            url = "https://blast.ncbi.nlm.nih.gov/blast/Blast.cgi?" + urllib.parse.urlencode(get_params)
            with urllib.request.urlopen(url) as get_res:
                result = get_res.read().decode('utf-8')
                
                if "Status=WAITING" in result:
                    continue
                if "Status=FAILED" in result or "Status=UNKNOWN" in result:
                    rename_file(f"{date_str}_{unique_num}_unknown.csv")
                    return
                if "Status=READY" in result or "<Hit_def>" in result:
                    match_name = re.search(r'<Hit_def>(.*?)</Hit_def>', result)
                    match_acc = re.search(r'<Hit_accession>(.*?)</Hit_accession>', result)
                    if match_name and match_acc:
                        protein_name = match_name.group(1).split()[0]
                        protein_name = re.sub(r'[^a-zA-Z0-9_]', '_', protein_name)
                        acc = match_acc.group(1)
                        rename_file(f"{date_str}_{acc}_{protein_name}.csv")
                        return
                    else:
                        break
        
        rename_file(f"{date_str}_{unique_num}_unknown.csv")
    except Exception as e:
        print(f"BLAST failed: {e}")
        rename_file(f"{date_str}_{unique_num}_unknown.csv")

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
    background_tasks: BackgroundTasks,
    intensities: List[str] = Form(...),
    calculate_csp: bool = Form(False)
):
    try:
        import time, uuid
        data_dir = os.path.join(BASE_DIR, "backend", "collected_data")
        os.makedirs(data_dir, exist_ok=True)
        file_name = f"data_{int(time.time())}_{str(uuid.uuid4())[:8]}.csv"
        file_path = os.path.join(data_dir, file_name)
        with open(file_path, "w", encoding="utf-8") as f:
            for item in intensities:
                f.write(item + "\n")
                
        residue_data = []
        feature_names = None
        aa_sequence = {}
        
        # If the frontend sends everything as ONE large string in a list, split it
        all_lines = []
        for item in intensities:
            all_lines.extend(item.replace('\r', '').split('\n'))
            
        # Parse lines
        valid_lines = [l.strip() for l in all_lines if l.strip()]
        if not valid_lines:
             raise HTTPException(status_code=400, detail="No data provided")
             
        def split_line(line):
            if ',' in line:
                return [p.strip() for p in line.split(',')]
            elif '\t' in line:
                return [p.strip() for p in line.split('\t')]
            else:
                return [p for p in re.split(r'\s+', line.strip()) if p]

        first_line = valid_lines[0]
        parts = split_line(first_line)
        
        # Check if first line is header (if 2nd col onwards are NOT floats)
        is_header = False
        if len(parts) >= 2:
            try:
                for v in parts[1:]:
                    if v and v.upper() not in ['NA', 'NAN']:
                        float(v)
            except ValueError:
                is_header = True
                feature_names = parts[1:]

                
        for s in valid_lines:
            if is_header and s == first_line:
                continue
                
            parts = split_line(s)
            if len(parts) < 1: continue
            
            res_id_str_raw = parts[0]
            # Strip non-numeric characters to get pure residue number (e.g. "A10" -> "10")
            res_id_numeric = re.sub(r'\D', '', res_id_str_raw)
            
            aa_match = re.match(r'^([A-Za-z]+)', res_id_str_raw)
            aa = aa_match.group(1).upper() if aa_match else ""
            
            if not res_id_numeric: 
                # If no number found, skip or keep raw? Let's skip to avoid NaN issues
                continue
                
            if aa:
                if len(aa) == 1:
                    aa_sequence[int(res_id_numeric)] = aa
                elif len(aa) == 3:
                    three_to_one = {'ALA':'A', 'ARG':'R', 'ASN':'N', 'ASP':'D', 'CYS':'C', 'GLN':'Q', 'GLU':'E', 'GLY':'G', 'HIS':'H', 'ILE':'I', 'LEU':'L', 'LYS':'K', 'MET':'M', 'PHE':'F', 'PRO':'P', 'SER':'S', 'THR':'T', 'TRP':'W', 'TYR':'Y', 'VAL':'V'}
                    if aa in three_to_one:
                        aa_sequence[int(res_id_numeric)] = three_to_one[aa]
                
            try:
                values = []
                for v in parts[1:]:
                    if not v or v.upper() in ['NA', 'NAN']:
                        values.append(np.nan)
                    else:
                        values.append(float(v))
                
                # If values are completely empty or all nan, we might want to still keep them if we do PPCA,
                # but if there's no data at all for this row except the ID, it might be useless.
                # We'll keep it. 
                
                # Use the processed numeric string as the ID
                residue_data.append([res_id_numeric] + values)
                
                # Fallback naming if no header
                if feature_names is None:
                     feature_names = [f"Feature_{i+1}" for i in range(len(values))]
            except ValueError:
                continue
        
        if not residue_data:
            raise HTTPException(status_code=400, detail="No valid residue data found")

        sequence_str = ""
        if aa_sequence:
            for res_id in sorted(aa_sequence.keys()):
                sequence_str += aa_sequence[res_id]
        
        background_tasks.add_task(run_blast_and_rename, file_path, sequence_str)

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