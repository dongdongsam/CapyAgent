import json
import pandas as pd

# The 302 (or so) standard hermaphrodite neurons
NEURON_NAMES = [
    "ADAL", "ADAR", "ADEL", "ADER", "ADFL", "ADFR", "ADLL", "ADLR", "AFDL", "AFDR",
    "AIAL", "AIAR", "AIBL", "AIBR", "AIML", "AIMR", "AINL", "AINR", "AIYL", "AIYR", "AIZL", "AIZR",
    "ALA", "ALML", "ALMR", "ALNL", "ALNR", "AQR",
    "AS1", "AS2", "AS3", "AS4", "AS5", "AS6", "AS7", "AS8", "AS9", "AS10", "AS11",
    "ASEL", "ASER", "ASGL", "ASGR", "ASHL", "ASHR", "ASIL", "ASIR", "ASJL", "ASJR", "ASKL", "ASKR",
    "AUAL", "AUAR", "AVAL", "AVAR", "AVBL", "AVBR", "AVDL", "AVDR", "AVEL", "AVER", "AVFL", "AVFR", "AVG", "AVHL", "AVHR", "AVJL", "AVJR", "AVKL", "AVKR", "AVL", "AVM",
    "AWAL", "AWAR", "AWBL", "AWBR", "AWCL", "AWCR",
    "BAGL", "BAGR", "BDUL", "BDUR", "CANL", "CANR",
    "DA1", "DA2", "DA3", "DA4", "DA5", "DA6", "DA7", "DA8", "DA9",
    "DB1", "DB2", "DB3", "DB4", "DB5", "DB6", "DB7",
    "DD1", "DD2", "DD3", "DD4", "DD5", "DD6",
    "DVA", "DVB", "DVC", "FLPL", "FLPR", "HSAL", "HSAR",
    "I1L", "I1R", "I2L", "I2R", "I3", "I4", "I5", "I6",
    "IL1DL", "IL1DR", "IL1L", "IL1R", "IL1VL", "IL1VR",
    "IL2DL", "IL2DR", "IL2L", "IL2R", "IL2VL", "IL2VR",
    "LUAL", "LUAR", "M1", "M2L", "M2R", "M3L", "M3R", "M4", "M5", "MCL", "MCR", "MI",
    "NSML", "NSMR", "OLLL", "OLLR", "OLQDL", "OLQDR", "OLQVL", "OLQVR",
    "PDA", "PDB", "PDC", "PDEL", "PDER", "PHAL", "PHAR", "PHBL", "PHBR", "PHCL", "PHCR", "PHDL", "PHDR",
    "PLML", "PLMR", "PLNL", "PLNR", "PQR",
    "PVCL", "PVCR", "PVDL", "PVDR", "PVEL", "PVER", "PVNL", "PVNR", "PVPL", "PVPR", "PVQL", "PVQR", "PVR", "PVT", "PVWL", "PVWR",
    "RIAL", "RIAR", "RIBL", "RIBR", "RICL", "RICR", "RID", "RIFL", "RIFR", "RIGL", "RIGR", "RIH", "RIML", "RIMR", "RIPL", "RIPR", "RIR", "RIS", "RIVL", "RIVR",
    "RMDDL", "RMDDR", "RMDL", "RMDR", "RMDVL", "RMDVR", "RMED", "RMEL", "RMER", "RMEV", "RMFL", "RMFR", "RMGL", "RMGR", "RMHL", "RMHR",
    "SAADL", "SAADR", "SAAVL", "SAAVR", "SABD", "SABVL", "SABVR", "SDQL", "SDQR",
    "SIADL", "SIADR", "SIAVL", "SIAVR", "SIBDL", "SIBDR", "SIBVL", "SIBVR",
    "SMBDL", "SMBDR", "SMBVL", "SMBVR", "SMDDL", "SMDDR", "SMDVL", "SMDVR",
    "URADL", "URADR", "URAVL", "URAVR", "URBL", "URBR", "URDL", "URDR", "URXL", "URXR", "URYDL", "URYDR", "URYVL", "URYVR",
    "VA1", "VA2", "VA3", "VA4", "VA5", "VA6", "VA7", "VA8", "VA9", "VA10", "VA11", "VA12",
    "VB1", "VB2", "VB3", "VB4", "VB5", "VB6", "VB7", "VB8", "VB9", "VB10", "VB11",
    "VC1", "VC2", "VC3", "VC4", "VC5", "VC6",
    "VD1", "VD2", "VD3", "VD4", "VD5", "VD6", "VD7", "VD8", "VD9", "VD10", "VD11", "VD12", "VD13"
]

# Ensure uniqueness
NEURON_NAMES = sorted(list(set(NEURON_NAMES)))
NUM_NEURONS = len(NEURON_NAMES)
print(f"Total unique neurons: {NUM_NEURONS}")

# Classification rules
SENSORY_PREFIXES = ["ADF", "ADL", "AFD", "ALM", "AQR", "ASE", "ASG", "ASH", "ASI", "ASJ", "ASK", "AWA", "AWB", "AWC", "BAG", "CEP", "FLP", "IL1", "IL2", "OLL", "OLQ", "PDA", "PQR", "PHD", "PHA", "PHB", "PLM", "PVD", "PVM", "SDQ", "URX", "URY", "ADE", "PDE"]
MOTOR_PREFIXES = ["AS", "DA", "DB", "DD", "M1", "M2", "M3", "M4", "M5", "RME", "SAA", "SAB", "SMB", "SMD", "URA", "URB", "VA", "VB", "VC", "VD", "HSN"]

def get_type(name):
    for prefix in SENSORY_PREFIXES:
        if name.startswith(prefix):
            return "sensory"
    for prefix in MOTOR_PREFIXES:
        if name.startswith(prefix):
            return "motor"
    return "interneuron"

neurons = []
for name in NEURON_NAMES:
    neurons.append({
        "name": name,
        "type": get_type(name)
    })

with open("data/neurons.json", "w") as f:
    json.dump(neurons, f, indent=4)

name_to_idx = {name: i for i, name in enumerate(NEURON_NAMES)}

chemical_adj = [[0.0] * NUM_NEURONS for _ in range(NUM_NEURONS)]
electrical_adj = [[0.0] * NUM_NEURONS for _ in range(NUM_NEURONS)]

df = pd.read_csv("data/connectome.csv")
df['Source'] = df['Source'].str.strip()
df['Target'] = df['Target'].str.strip()

for _, row in df.iterrows():
    src = row['Source']
    tgt = row['Target']
    weight = float(row['Weight'])
    conn_type = row['Type'].strip()
    
    if src in name_to_idx and tgt in name_to_idx:
        i, j = name_to_idx[src], name_to_idx[tgt]
        if conn_type == 'chemical':
            chemical_adj[i][j] = weight
        elif conn_type == 'electrical':
            electrical_adj[i][j] = weight
            electrical_adj[j][i] = weight

with open("data/connectome.json", "w") as f:
    json.dump({
        "chemical": chemical_adj,
        "electrical": electrical_adj
    }, f)

print("Setup completed successfully.")
