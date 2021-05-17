import os
import sys
import time
import warnings

from openea.modules.args.args_hander import load_args
from openea.modules.load.kgs import read_kgs_from_folder
from openea.models.basic_model import BasicModel
from openea.approaches.ontoea import OntoEA


warnings.filterwarnings('ignore')


class ModelFamily(object):
    BasicModel = BasicModel
    OntoEA = OntoEA


def get_model(model_name):
    return getattr(ModelFamily, model_name)


if __name__ == '__main__':
    t = time.time()
    args = load_args(sys.argv[1])
    args.training_data = args.training_data + sys.argv[2] + '/'
    # args.dataset_division = sys.argv[3]
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[3]
    args.lang = os.path.split(sys.argv[2])[-1]
    # print(sys.argv[2],args.lang)

    print(args.embedding_module)
    print(args)
    remove_unlinked = False
    if args.embedding_module == "RSN4EA":
        remove_unlinked = True

    if args.onto_valid:
        kgs = read_kgs_from_folder(args.training_data, args.dataset_division,
                                   args.alignment_module, args.ordered, args.lang,
                                   remove_unlinked=remove_unlinked,
                                   onto_mode=args.cvlink_module,
                                   seed_ent_type=args.seed_ent_type,
                                   check_version=args.check_version,
                                   data_version=args.data_version)
    else:
        kgs = read_kgs_from_folder(args.training_data, args.dataset_division,
                                   args.alignment_module, args.ordered, args.lang,
                                   remove_unlinked=remove_unlinked,
                                   data_version=args.data_version)
    model = get_model(args.embedding_module)()
    model.set_args(args)
    model.set_kgs(kgs)
    model.init()
    model.run()
    model.test()
    model.save()
    print("Total run time = {:.3f} s.".format(time.time() - t))
