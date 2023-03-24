import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms

NORM = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
te_transforms = transforms.Compose([transforms.Resize((32, 32)),
									transforms.ToTensor(),
									transforms.Normalize((0.1307,), (0.3081,))])
mnist_transforms = transforms.Compose([transforms.Resize((32, 32)),
										transforms.RandomHorizontalFlip(),
										transforms.ToTensor(),
										 transforms.Normalize((0.1307,), (0.3081,))])
tr_transforms = transforms.Compose([transforms.RandomCrop(32, padding=4),
									transforms.RandomHorizontalFlip(),
									transforms.ToTensor(),
									transforms.Normalize(*NORM)])


common_corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
					'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
					'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']

def prepare_test_data(args):
	if args.dataset == 'mnist':
		tesize = 10000
		if not hasattr(args, 'corruption') or args.corruption == 'original':
			print('Test on the original test set')
			teset = torchvision.datasets.MNIST(root=args.dataroot,
												train=False, download=True, transform=te_transforms)
	else:
		raise Exception('Dataset not found!')
	teloader = torch.utils.data.DataLoader(teset, batch_size=args.batch_size,
											shuffle=False, num_workers=0)
	return teset, teloader

def prepare_train_data(args):
	print('Preparing data...')
	if args.dataset == 'mnist':
		trset = torchvision.datasets.MNIST(root=args.dataroot,
										train=True, download=True, transform=mnist_transforms)
	else:
		raise Exception('Dataset not found!')
	trloader = torch.utils.data.DataLoader(trset, batch_size=args.batch_size,
											shuffle=True, num_workers=0)
	return trset, trloader
	