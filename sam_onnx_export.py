import logging
from pathlib import Path


logging.basicConfig(level=logging.INFO)


def file_exists(filepath: Path) -> bool:
    my_file = Path(filepath)
    logging.info(f"file_exists:{filepath}?")
    try:
        my_file.resolve(strict=True)
        logging.info(f"file at path {filepath} exists and is a file!")
        return my_file.is_file()
    except FileNotFoundError as fnfe:
        logging.error("fnfe:{fnfe}")
        raise fnfe


if __name__ == '__main__':
    # pip install git+https://github.com/facebookresearch/segment-anything.git
    # pip install mobile_sam
    import os
    from dotenv import load_dotenv

    from mobile_sam.build_sam import sam_model_registry
    from huggingface_hub import hf_hub_download
    from segment_anything.utils.onnx import SamOnnxModel
    import torch

    load_dotenv()

    # https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables
    # set the HF_HOME env variable
    # ROOT = Path(__file__).resolve().parent
    # logging.info(f"root:{ROOT}.")
    # os.environ["HF_HOME"] = str(ROOT / "original_models")
    ROOT = Path(__file__).resolve().parent
    models_folder = ROOT / "original_models"
    logging.info(f"models_folder:{models_folder}#")
    # Load SAM model
    hf_reponame_model = os.getenv("HF_MODEL_REPO")
    hf_filename_model = os.getenv("HF_MODEL_FILENAME")
    model_prefix = os.getenv("HF_MODEL_PREFIX_FILENAME")
    logging.info(f"start download {model_prefix} => {hf_filename_model} to {models_folder} folder...")
    hf_hub_download(repo_id=hf_reponame_model, filename=hf_filename_model, local_dir=models_folder)
    pytorch_model = f"{models_folder}/{hf_filename_model}"
    logging.info(f"downloaded: {pytorch_model} ...")
    
    file_exists(pytorch_model)
    
    sam = sam_model_registry["vit_t"](checkpoint=pytorch_model)

    logging.info(f"registered {pytorch_model}...")
    encoder_onnx = f"{model_prefix}.encoder.onnx"

    # Export images encoder from SAM model to ONNX
    logging.info(f"start export encoder: {encoder_onnx} ...")
    torch.onnx.export(
        f=encoder_onnx,
        model=sam.image_encoder,
        args=torch.randn(1, 3, 1024, 1024),
        input_names=["images"],
        output_names=["embeddings"],
        export_params=True
    )
    file_exists(ROOT / encoder_onnx)

    # Export mask decoder from SAM model to ONNX
    onnx_model = SamOnnxModel(sam, return_single_mask=True)
    embed_dim = sam.prompt_encoder.embed_dim
    embed_size = sam.prompt_encoder.image_embedding_size
    mask_input_size = [4 * x for x in embed_size]
    dummy_inputs = {
        "image_embeddings": torch.randn(1, embed_dim, *embed_size, dtype=torch.float),
        "point_coords": torch.randint(low=0, high=1024, size=(1, 5, 2), dtype=torch.float),
        "point_labels": torch.randint(low=0, high=4, size=(1, 5), dtype=torch.float),
        "mask_input": torch.randn(1, 1, *mask_input_size, dtype=torch.float),
        "has_mask_input": torch.tensor([1], dtype=torch.float),
        "orig_im_size": torch.tensor([1500, 2250], dtype=torch.float),
    }
    decoder_onnx = f"{model_prefix}.decoder.onnx"
    logging.info(f"start export: {decoder_onnx} ...")
    output_names = ["masks", "iou_predictions", "low_res_masks"]
    torch.onnx.export(
        f=decoder_onnx,
        model=onnx_model,
        args=tuple(dummy_inputs.values()),
        input_names=list(dummy_inputs.keys()),
        output_names=output_names,
        dynamic_axes={
            "point_coords": {1: "num_points"},
            "point_labels": {1: "num_points"}
        },
        export_params=True,
        opset_version=17,
        do_constant_folding=True
    )
    file_exists(ROOT / decoder_onnx)
    logging.info("ok4")
