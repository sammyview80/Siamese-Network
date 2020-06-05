from main import Main
from face import live


def app():
    choice = input("Do You want to Train Model:")
    if choice == 1 or choice == 'yes' or choice == 'y' or choice == 'Y':
        Main().run()
    else:
        print('Running Live...')
        live()


if __name__ == "__main__":
    app()