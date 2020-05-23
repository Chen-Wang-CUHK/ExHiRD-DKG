import os
import argparse

parser = argparse.ArgumentParser(description='dicoqg saved_model postprocess')
parser.add_argument('--model_dir', '-model_dir', type=str, default='saved_models',
                    help='The directory of the saved models')
# parser.add_argument('--model_name', type=str, required=True,
#                     help='The directory of the saved models')
args = parser.parse_args()


def select_best_remove_others(args):
    # Only remain the best ppl and acc models and remove the other models for a set of hyperparameters
    model_list = [f for f in os.listdir(args.model_dir) if os.path.isfile(os.path.join(args.model_dir, f))]
    model_best = {}
    for model in model_list:
        # all the models have the same pattern like:
        # seqE1_seqD1_dp0.0_TitanV_PbfA_ordered_addSemicolon_removedAPSep_addBiSTokens_ppl_5.03_acc_63.98_step_142500.pt
        assert 'ppl' in model
        assert 'acc' in model
        ppl_splitter = 'ppl'
        acc_splitter = 'acc'

        model_name_base = model.split('_' + ppl_splitter)[0]
        ppl = float(model.split(ppl_splitter + '_')[-1].split('_')[0])
        acc = float(model.split(acc_splitter + '_')[-1].split('_')[0])
        step = int(model.split('step_')[-1].split('.')[0])
        cur_model_stats = {'ppl': ppl, 'acc': acc, 'step': step}
        removed = None
        if model_name_base in model_best:
            cur_best_model_triplet = model_best[model_name_base][0]
            # 1. choose the model with lower ppl
            if ppl < cur_best_model_triplet['ppl']:
                removed = model_best[model_name_base][1]
                model_best[model_name_base] = (cur_model_stats, model)
            elif ppl > cur_best_model_triplet['ppl']:
                removed = model
            else:
                # if the ppl is the same, choose the model with higher acc
                if acc > cur_best_model_triplet['acc']:
                    removed = model_best[model_name_base][1]
                    model_best[model_name_base] = (cur_model_stats, model)
                elif acc < cur_best_model_triplet['acc']:
                    removed = model
                else:
                    # if the ppl and acc are the same, choose the model with smaller step
                    if step < cur_best_model_triplet['step']:
                        removed = model_best[model_name_base][1]
                        model_best[model_name_base] = (cur_model_stats, model)
                    else:
                        removed = model
        else:
            model_best[model_name_base] = (cur_model_stats, model)
            removed = None

        if removed is not None:
            # seed3435.s2s_h200_acc_54.541_ppl_10.365_epoch6_step16000.pt
            removed_file = removed
            print('Remove {}'.format(os.path.join(args.model_dir, removed_file)))
            os.remove(os.path.join(args.model_dir, removed_file))


if __name__ == '__main__':
    select_best_remove_others(args)