if not os.path.isdir(image_dir) or not any(f.lower().endswith('.tif') for f in os.listdir(image_dir)):
    print(f"Error: Directory '{image_dir}' not found or contains no .tif files.")
    print("Please update the 'image_dir' variable to point to your dataset folder.")
else:
    train_dataset = MedicalImageDataset(image_dir=image_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    print(f"Successfully loaded {len(train_dataset)} images.")

    # --- Instantiate model, loss, and optimizer ---
    model = JInv_UNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # ==============================================================================
    # 5. MAIN TRAINING LOOP
    # ==============================================================================
    print("\nStarting model training...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for batch_images in train_loader:
            batch_images = batch_images.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch_images)
            loss = criterion(outputs, batch_images)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * batch_images.size(0)
            
        epoch_loss = running_loss / len(train_dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.6f}")
        
    end_time = time.time()
    print(f"Training finished in {(end_time - start_time)/60:.2f} minutes.")

    # ==============================================================================
    # 6. VISUALIZE RESULTS
    # ==============================================================================
    print("\nVisualizing results on a sample batch...")
    model.eval()
    with torch.no_grad():
        # Get one batch of images from the loader
        sample_images = next(iter(train_loader)).to(device)
        
        # Denoise the batch
        denoised_images = model(sample_images)

        # Move tensors to CPU for plotting
        sample_images = sample_images.cpu().numpy()
        denoised_images = denoised_images.cpu().numpy()

        # Display the first few images from the batch
        n_images_to_show = min(batch_size, 4)
        fig, axes = plt.subplots(2, n_images_to_show, figsize=(15, 8))
        for i in range(n_images_to_show):
            # Original Noisy Image
            axes[0, i].imshow(np.squeeze(sample_images[i]), cmap='gray')
            axes[0, i].set_title("Original Noisy")
            axes[0, i].axis('off')
            
            # Denoised Image
            axes[1, i].imshow(np.squeeze(denoised_images[i]), cmap='gray')
            axes[1, i].set_title("Denoised by N2I")
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.show()