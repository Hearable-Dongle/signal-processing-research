from siamese_unet import SiameseUnet

def main():
    model = SiameseUnet()
    total_params = sum(param.numel() for param in model.parameters())
    print("Total number of params:", total_params)

if __name__ == "__main__":
    main()

