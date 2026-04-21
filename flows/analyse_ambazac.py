import onecode

def run():
    # TEST SIMPLE : Juste un slider et un texte
    nom = onecode.text_input("ton_nom", "Explorateur", label="Quel est ton nom ?")
    valeur = onecode.slider("test_slider", 10.0, min=0.0, max=100.0, label="Curseur de Test")
    
    onecode.Logger.info(f"Bonjour {nom}, le curseur est sur {valeur}")
