# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import argparse
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
import os
import torch
from torch.utils.data import DataLoader
import torchaudio
from hyperpyyaml import load_hyperpyyaml
from tqdm import tqdm
from cosyvoice.cli.model import CosyVoiceModel, CosyVoice2Model
from cosyvoice.dataset.dataset import Dataset


def get_args():
    parser = argparse.ArgumentParser(description='inference with your model')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--prompt_data', required=True, help='prompt data file')
    parser.add_argument('--prompt_utt2data', required=True, help='prompt data file')
    parser.add_argument('--tts_text', required=True, help='tts input file')
    parser.add_argument('--qwen_pretrain_path', required=False, help='qwen pretrain path')
    parser.add_argument('--llm_model', required=True, help='llm model file')
    parser.add_argument('--flow_model', required=True, help='flow model file')
    parser.add_argument('--hifigan_model', required=True, help='hifigan model file')
    parser.add_argument('--gpu',
                        type=int,
                        default=-1,
                        help='gpu id for this rank, -1 for cpu')
    parser.add_argument('--mode',
                        default='sft',
                        choices=['sft', 'zero_shot'],
                        help='inference mode')
    parser.add_argument('--result_dir', required=True, help='asr result file')
    args = parser.parse_args()
    print(args)
    return args


def main():
    args = get_args()
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # Init cosyvoice models from configs
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    try:
        with open(args.config, 'r') as f:
            configs = load_hyperpyyaml(f, overrides={'qwen_pretrain_path': args.qwen_pretrain_path})
        model = CosyVoice2Model(configs['llm'], configs['flow'], configs['hift'])
    except Exception:
        try:
            with open(args.config, 'r') as f:
                configs = load_hyperpyyaml(f)
            model = CosyVoiceModel(configs['llm'], configs['flow'], configs['hift'])
        except Exception:
            raise TypeError('no valid model_type!')

    model.load(args.llm_model, args.flow_model, args.hifigan_model)

    test_dataset = Dataset(args.prompt_data, data_pipeline=configs['data_pipeline'], mode='inference', shuffle=False, partition=False,
                           tts_file=args.tts_text, prompt_utt2data=args.prompt_utt2data)
    test_data_loader = DataLoader(test_dataset, batch_size=None, num_workers=0)

    sample_rate = configs['sample_rate']
    del configs
    os.makedirs(args.result_dir, exist_ok=True)
    fn = os.path.join(args.result_dir, 'wav.scp')
    f = open(fn, 'w')
    with torch.no_grad():
        for _, batch in tqdm(enumerate(test_data_loader)):
            utts = batch["utts"]
            assert len(utts) == 1, "inference mode only support batchsize 1"
            text_token = batch["text_token"].to(device)
            text_token_len = batch["text_token_len"].to(device)
            tts_index = batch["tts_index"]
            tts_text_token = batch["tts_text_token"].to(device)
            tts_text_token_len = batch["tts_text_token_len"].to(device)
            speech_token = batch["speech_token"].to(device)
            speech_token_len = batch["speech_token_len"].to(device)
            speech_feat = batch["speech_feat"].to(device)
            speech_feat_len = batch["speech_feat_len"].to(device)
            utt_embedding = batch["utt_embedding"].to(device)
            spk_embedding = batch["spk_embedding"].to(device)
            if args.mode == 'sft':
                model_input = {'text': tts_text_token, 'text_len': tts_text_token_len,
                               'llm_embedding': spk_embedding, 'flow_embedding': spk_embedding}
            else:
                model_input = {'text': tts_text_token, 'text_len': tts_text_token_len,
                               'prompt_text': text_token, 'prompt_text_len': text_token_len,
                               'llm_prompt_speech_token': speech_token, 'llm_prompt_speech_token_len': speech_token_len,
                               'flow_prompt_speech_token': speech_token, 'flow_prompt_speech_token_len': speech_token_len,
                               'prompt_speech_feat': speech_feat, 'prompt_speech_feat_len': speech_feat_len,
                               'llm_embedding': utt_embedding, 'flow_embedding': utt_embedding}
            tts_speeches = []
            for model_output in model.tts(**model_input):
                tts_speeches.append(model_output['tts_speech'])
            tts_speeches = torch.concat(tts_speeches, dim=1)
            tts_key = '{}_{}'.format(utts[0], tts_index[0])
            tts_fn = os.path.join(args.result_dir, '{}.wav'.format(tts_key))
            torchaudio.save(tts_fn, tts_speeches, sample_rate=sample_rate, backend='soundfile')
            f.write('{} {}\n'.format(tts_key, tts_fn))
            f.flush()
    f.close()
    logging.info('Result wav.scp saved in {}'.format(fn))


if __name__ == '__main__':
    main()
