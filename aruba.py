def format_aruba():
    f = open("data/data_aruba_carlos.txt", "r")
    content = f.read()

    content = content.replace("\r", " ")
    content = content.replace("\t", " ")
    content = content.replace("  ", " ")
    content = content.replace(" ", "\t")

    write_file = open("data/data_aruba_formatted.txt", "w")
    write_file.write(content)
