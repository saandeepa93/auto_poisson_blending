
import click
from sys import exit as e

from modules.copy_paste import combine_img
from modules.util import get_config


@click.command()
@click.option('--config', help="path to config file")
def blend(config):
  configs = get_config(config)
  combine_img(configs)



@click.group()
def main():
  pass


if __name__ == '__main__':
  main.add_command(blend)
  main()
