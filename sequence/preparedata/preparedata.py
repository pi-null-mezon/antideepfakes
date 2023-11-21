import subprocess


def config(dthresh: float, strobe: int, tside: int, teyes: int, vshift: float):
    options = {
        'face_detector_input_size': 200,
        'face_detector_threshold': dthresh,
        'target_width': tside,
        'target_height': tside,
        'extraction_mode': 'EYES',
        'target_eyes_dst': teyes,
        'vertical_shift': vshift,
        'rotate_eyes': True,
        'without_annotation': True,
        'output_codec': 'jpg',
        'frames_strobe': strobe,
        'naming_mode': 'PRESERVE'
    }
    args = []
    for key in options:
        if isinstance(options[key], bool):
            if options[key]:
                args.append(f"--{key}")
        else:
            args.append(f"--{key}")
            args.append(str(options[key]))
    return args


def spawn(source_dir: str, target_dir: str, videos: bool, args: list):
    if videos:
        cmd = ["python", "-m", "ovfacetools.facextractor", "--videos_dir", source_dir, "--output_dir", target_dir] + args
    else:
        cmd = ["python", "-m", "ovfacetools.facextractor", "--photos_dir", source_dir, "--output_dir", target_dir] + args
    p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL)
    print(f" - process {p.pid} started: '{source_dir}' >> '{target_dir}'")
    return p


crop_size = 256  # 224
iod_dst = 60     # 90
vshift = -0.1    # -0.2

crop_format = f"{crop_size}x{iod_dst}x{-vshift}"
target_location = f"/home/alex/Fastdata/deepfakes/sequence/{crop_format}"

procs = list()
conf = config(dthresh=0.9, strobe=7, tside=crop_size, teyes=iod_dst, vshift=vshift)

'''dataset_name = 'roop'
procs.append(spawn(f"/media/alex/HDD1_2T/Deepfakes/{dataset_name}/moves/live", f"{target_location}/{dataset_name}/live", True, conf))
procs.append(spawn(f"/media/alex/HDD1_2T/Deepfakes/{dataset_name}/moves/fake", f"{target_location}/{dataset_name}/fake", True, conf))
procs.append(spawn(f"/media/alex/HDD1_2T/Deepfakes/{dataset_name}/nums/live", f"{target_location}/{dataset_name}/live", True, conf))
procs.append(spawn(f"/media/alex/HDD1_2T/Deepfakes/{dataset_name}/nums/fake", f"{target_location}/{dataset_name}/fake", True, conf))
for proc in procs:
    proc.wait()
    print(f" - process {proc.pid} finished, return code: {proc.returncode}")
procs = list()

dataset_name = 'Celeb-DF-v2'
procs.append(spawn(f"/media/alex/HDD1_2T/Deepfakes/{dataset_name}/Celeb-real", f"{target_location}/{dataset_name}/live", True, conf))
procs.append(spawn(f"/media/alex/HDD1_2T/Deepfakes/{dataset_name}/YouTube-real", f"{target_location}/{dataset_name}/live", True, conf))
procs.append(spawn(f"/media/alex/HDD1_2T/Deepfakes/{dataset_name}/Celeb-synthesis", f"{target_location}/{dataset_name}/fake", True, conf))
for proc in procs:
    proc.wait()
    print(f" - process {proc.pid} finished, return code: {proc.returncode}")
procs = list()

dataset_name = 'FaceForensics++'
procs.append(spawn(f"/media/alex/HDD1_2T/Deepfakes/{dataset_name}/original_sequences/actors/c23/videos", f"{target_location}/{dataset_name}/live", True, conf))
procs.append(spawn(f"/media/alex/HDD1_2T/Deepfakes/{dataset_name}/original_sequences/youtube/c23/videos", f"{target_location}/{dataset_name}/live", True, conf))
procs.append(spawn(f"/media/alex/HDD1_2T/Deepfakes/{dataset_name}/manipulated_sequences/DeepFakeDetection/c23/videos", f"{target_location}/{dataset_name}/fake", True, conf))
procs.append(spawn(f"/media/alex/HDD1_2T/Deepfakes/{dataset_name}/manipulated_sequences/Deepfakes/c23/videos", f"{target_location}/{dataset_name}/fake", True, conf))
procs.append(spawn(f"/media/alex/HDD1_2T/Deepfakes/{dataset_name}/manipulated_sequences/Face2Face/c23/videos", f"{target_location}/{dataset_name}/fake", True, conf))
procs.append(spawn(f"/media/alex/HDD1_2T/Deepfakes/{dataset_name}/manipulated_sequences/FaceShifter/c23/videos", f"{target_location}/{dataset_name}/fake", True, conf))
procs.append(spawn(f"/media/alex/HDD1_2T/Deepfakes/{dataset_name}/manipulated_sequences/FaceSwap/c23/videos", f"{target_location}/{dataset_name}/fake", True, conf))
#procs.append(spawn(f"/media/alex/HDD1_2T/Deepfakes/{dataset_name}/manipulated_sequences/NeuralTextures/c23/videos", f"{target_location}/{dataset_name}/fake", True, conf))
for proc in procs:
    proc.wait()
    print(f" - process {proc.pid} finished, return code: {proc.returncode}")
procs = list()

dataset_name = "dfdc"
for part in ['0', '1', '3', '5', '07', '11']:
    procs.append(spawn(f"/media/alex/HDD1_2T/Deepfakes/{dataset_name}/train_part_{part}/live", f"{target_location}/{dataset_name}/live", True, conf))
    procs.append(spawn(f"/media/alex/HDD1_2T/Deepfakes/{dataset_name}/train_part_{part}/fake", f"{target_location}/{dataset_name}/fake", True, conf))
for proc in procs:
    proc.wait()
    print(f" - process {proc.pid} finished, return code: {proc.returncode}")
procs = list()
procs.append(spawn("/media/alex/Datastorage/Testdata/Video/distortion_liveness/all/live", f"{target_location}/{dataset_name}/live", True, conf))
for part in ['17', '23', '41', '43', '47']:
    procs.append(spawn(f"/media/alex/HDD1_2T/Deepfakes/{dataset_name}/train_part_{part}/live", f"{target_location}/{dataset_name}/live", True, conf))
    procs.append(spawn(f"/media/alex/HDD1_2T/Deepfakes/{dataset_name}/train_part_{part}/fake", f"{target_location}/{dataset_name}/fake", True, conf))'''


dataset_name = "dfdc"
for part in ['13', '33']:
    procs.append(spawn(f"/media/alex/HDD1_2T/Deepfakes/{dataset_name}/train_part_{part}/live", f"{target_location}/{dataset_name}_test/live", True, conf))
    procs.append(spawn(f"/media/alex/HDD1_2T/Deepfakes/{dataset_name}/train_part_{part}/fake", f"{target_location}/{dataset_name}_test/fake", True, conf))

for proc in procs:
    proc.wait()
    print(f" - process {proc.pid} finished, return code: {proc.returncode}")
print("All processes has been finished")
