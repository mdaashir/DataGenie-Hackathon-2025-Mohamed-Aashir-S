from Backend.Routes import router

@router.get("/")
def home():
    return "Hello World"