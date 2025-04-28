import glob, os, numpy as np, webdataset as wds, json, re
ROOT = "data/cubemap_latents_dir"      # existing *.pt or *.npy files
CAPT = "data/polyhaven_captions.jsonl" # your caption file
OUT  = "data/polyhaven_latents.tar"

caps = {j["id"]: j["caption"] for j in map(json.loads, open(CAPT))}
wr   = wds.TarWriter(OUT, encoder=False)

for f in glob.glob(f"{ROOT}/*_lat.pt"):           # or *.npy
    pano_id = re.match(r".*/(.*)_lat\.pt", f).group(1)
    wr.write({"__key__": pano_id,
              "lat.pt": np.load(f),
              "txt":    caps[pano_id]})
wr.close()