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
        'frames_strobe': strobe
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
target_location = f"/home/alex/Fastdata/deepfakes/singleshot/{crop_format}"

procs = list()
conf = config(dthresh=0.9, strobe=13, tside=crop_size, teyes=iod_dst, vshift=vshift)

dataset_name = 'Celeb-DF-v2'
procs.append(spawn(f"/media/alex/HDD1_2T/Deepfakes/{dataset_name}/Celeb-real", f"{target_location}/{dataset_name}/live", True, conf))
procs.append(spawn(f"/media/alex/HDD1_2T/Deepfakes/{dataset_name}/YouTube-real", f"{target_location}/{dataset_name}/live", True, conf))
procs.append(spawn(f"/media/alex/HDD1_2T/Deepfakes/{dataset_name}/Celeb-synthesis", f"{target_location}/{dataset_name}/fake", True, conf))

dataset_name = 'FaceForensics++'
procs.append(spawn(f"/media/alex/HDD1_2T/Deepfakes/{dataset_name}/original_sequences/actors", f"{target_location}/{dataset_name}/live", True, conf))
procs.append(spawn(f"/media/alex/HDD1_2T/Deepfakes/{dataset_name}/original_sequences/youtube", f"{target_location}/{dataset_name}/live", True, conf))
procs.append(spawn(f"/media/alex/HDD1_2T/Deepfakes/{dataset_name}/manipulated_sequences/DeepFakeDetection", f"{target_location}/{dataset_name}/fake", True, conf))
procs.append(spawn(f"/media/alex/HDD1_2T/Deepfakes/{dataset_name}/manipulated_sequences/Deepfakes", f"{target_location}/{dataset_name}/fake", True, conf))
procs.append(spawn(f"/media/alex/HDD1_2T/Deepfakes/{dataset_name}/manipulated_sequences/Face2Face", f"{target_location}/{dataset_name}/fake", True, conf))
procs.append(spawn(f"/media/alex/HDD1_2T/Deepfakes/{dataset_name}/manipulated_sequences/FaceShifter", f"{target_location}/{dataset_name}/fake", True, conf))
procs.append(spawn(f"/media/alex/HDD1_2T/Deepfakes/{dataset_name}/manipulated_sequences/FaceSwap", f"{target_location}/{dataset_name}/fake", True, conf))
procs.append(spawn(f"/media/alex/HDD1_2T/Deepfakes/{dataset_name}/manipulated_sequences/NeuralTextures", f"{target_location}/{dataset_name}/fake", True, conf))
procs.append(spawn(f"/media/alex/Biosamples/Toloka/Four_live_samples", f"{target_location}/{dataset_name}/live", True, conf))
procs.append(spawn(f"/media/alex/Datastorage/Testdata/Video/distortion_liveness/all/live", f"{target_location}/{dataset_name}/live", True, conf))

dataset_name = 'dfdc/train_part_07'
procs.append(spawn(f"/media/alex/HDD1_2T/Deepfakes/{dataset_name}/live", f"{target_location}/{dataset_name}/live", True, conf))
procs.append(spawn(f"/media/alex/HDD1_2T/Deepfakes/{dataset_name}/fake", f"{target_location}/{dataset_name}/fake", True, conf))
procs.append(spawn("/media/alex/Datastorage/Testdata/Faces/RAW/Geodata_smartphone", f"{target_location}/{dataset_name}/live", False, conf))
procs.append(spawn("/media/alex/Datastorage/Testdata/Faces/RAW/Maximus", f"{target_location}/{dataset_name}/live", False, conf))

dataset_name = 'dfdc/train_part_17'
procs.append(spawn(f"/media/alex/HDD1_2T/Deepfakes/{dataset_name}/live", f"{target_location}/{dataset_name}/live", True, conf))
procs.append(spawn(f"/media/alex/HDD1_2T/Deepfakes/{dataset_name}/fake", f"{target_location}/{dataset_name}/fake", True, conf))
procs.append(spawn("/media/alex/Datastorage/Testdata/Faces/RAW/RTK", f"{target_location}/{dataset_name}/live", False, conf))
procs.append(spawn("/media/alex/Datastorage/Testdata/Faces/RAW/NMZ", f"{target_location}/{dataset_name}/live", False, conf))

dataset_name = 'dfdc/train_part_43'
procs.append(spawn(f"/media/alex/HDD1_2T/Deepfakes/{dataset_name}/live", f"{target_location}/{dataset_name}/live", True, conf))
procs.append(spawn(f"/media/alex/HDD1_2T/Deepfakes/{dataset_name}/fake", f"{target_location}/{dataset_name}/fake", True, conf))
procs.append(spawn("/media/alex/Datastorage/Testdata/Faces/RAW/Toloka", f"{target_location}/{dataset_name}/live", False, conf))
procs.append(spawn("/media/alex/Datastorage/Testdata/Faces/RAW/Fullface", f"{target_location}/{dataset_name}/live", False, conf))
procs.append(spawn("/media/alex/Datastorage/Testdata/Faces/RAW/EBS", f"{target_location}/{dataset_name}/live", False, conf))

for part in ['0', '1', '3', '5', '11', '23', '41', '47']:
    dataset_name = f'dfdc/train_part_{part}'
    procs.append(spawn(f"/media/alex/HDD1_2T/Deepfakes/{dataset_name}/live", f"{target_location}/{dataset_name}/live", True, conf))
    procs.append(spawn(f"/media/alex/HDD1_2T/Deepfakes/{dataset_name}/fake", f"{target_location}/{dataset_name}/fake", True, conf))

for proc in procs:
    proc.wait()
    print(f" - process {proc.pid} finished, return code: {proc.returncode}")
print("All processes has been finished")
