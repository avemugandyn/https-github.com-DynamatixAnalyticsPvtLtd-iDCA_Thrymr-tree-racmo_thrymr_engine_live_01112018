from web.app import create_app

app,model = create_app()
if __name__ == '__main__':
    app.run(port=1231, debug=False, host="127.0.0.1")