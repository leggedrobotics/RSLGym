from shutil import copyfile
import datetime
import os
import ntpath
import subprocess
import argparse

class ConfigurationSaver:
    def __init__(self, log_dir, save_items, args, save_git=True):
        self._data_dir = log_dir + '/' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        os.makedirs(self._data_dir)

        if save_items is not None:
            for save_item in save_items:
                base_file_name = ntpath.basename(save_item)
                copyfile(save_item, self._data_dir + '/' + base_file_name)

        if isinstance(args, argparse.Namespace):
            self.save_arg_parser(self._data_dir, args)

        if save_git:
            self.save_git_information(self._data_dir)

    @property
    def data_dir(self):
        return self._data_dir

    def save_git_information(self, outdir):
        # Save `git rev-parse HEAD` (SHA of the current commit)
        with open(os.path.join(outdir, "git-head.txt"), "wb") as f:
            f.write(subprocess.check_output("git rev-parse HEAD".split()))

        with open(os.path.join(outdir, "git-submodules.txt"), "wb") as f:
            f.write(subprocess.check_output("git submodule".split()))

        # Save `git status`
        with open(os.path.join(outdir, "git-status.txt"), "wb") as f:
            f.write(subprocess.check_output("git status".split()))

        # Save `git log`
        with open(os.path.join(outdir, "git-log.txt"), "wb") as f:
            f.write(subprocess.check_output("git log".split()))

        # Save `git diff`
        with open(os.path.join(outdir, "git-diff.txt"), "wb") as f:
            f.write(subprocess.check_output("git diff --submodule=diff".split()))

    def save_arg_parser(self, outdir, args):
        with open(os.path.join(outdir, "args.txt"), "w") as f:
            for key, value in args.__dict__.items():
                f.write(key + ': ' + str(value) + '\n')


def TensorboardLauncher(directory_path):
    from tensorboard import program
    import webbrowser
    # learning visualizer
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', directory_path])
    url = tb.launch()
    print("[RAISIM_GYM] Tensorboard session created: "+url)
    webbrowser.open_new(url)
