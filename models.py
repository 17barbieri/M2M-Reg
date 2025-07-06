import itk
import logging
import numpy as np
import torch
import icon_registration as icon
import icon_registration.network_wrappers as network_wrappers
import icon_registration.networks as networks
from icon_registration import config
from icon_registration.losses import ICONLoss, ICONLoss_can, ICONLoss_can_mono, flips
from icon_registration.mermaidlite import compute_warped_image_multiNC
import icon_registration.itk_wrapper
import ml_collections

class GradientICONSparse_M2M(network_wrappers.RegistrationModule):
    def __init__(self, args, network, similarity, use_label=False):

        super().__init__()

        self.regis_net = network
        self.lambda_inv = args.lambda_inv
        self.lambda_can = args.lambda_can
        self.similarity = similarity
        self.use_label = use_label
        self.input_shape = args.input_shape
        self.num_cano = args.num_cano
        self.log_mono = args.log_mono

    def create_Iepsilon(self):
        noise = 2 * torch.randn(*self.identity_map.shape).to(config.device) / self.identity_map.shape[-1]
        if len(self.input_shape) - 2 == 3:
            return (self.identity_map + noise)[:, :, ::2, ::2, ::2]
        elif len(self.input_shape) - 2 == 2:
            return (self.identity_map + noise)[:, :, ::2, ::2]
        return self.identity_map + noise

    def get_direction_vectors(self, delta=0.001):
        if len(self.identity_map.shape) == 4:
            return (torch.tensor([[[[delta]], [[0.0]]]]).to(config.device),
                    torch.tensor([[[[0.0]], [[delta]]]]).to(config.device))
        elif len(self.identity_map.shape) == 5:
            return (torch.tensor([[[[[delta]]], [[[0.0]]], [[[0.0]]]]]).to(config.device),
                    torch.tensor([[[[[0.0]]], [[[delta]]], [[[0.0]]]]]).to(config.device),
                    torch.tensor([[[[0.0]]], [[[0.0]]], [[[delta]]]]).to(config.device))
        return (torch.tensor([[[delta]]]).to(config.device),)

    def forward(self, image_A, image_B, cano_A, cano_B, label_A=None, label_B=None, dice_logging=False):
        assert self.identity_map.shape[2:] == image_A.shape[2:]
        assert self.identity_map.shape[2:] == image_B.shape[2:]
        assert self.identity_map.shape[2:] == cano_A.shape[2:]
        assert self.identity_map.shape[2:] == cano_B.shape[2:]

        self.identity_map.isIdentity = True
        self.phi_AB = self.regis_net(image_A, image_B)
        self.phi_AB_vectorfield = self.phi_AB(self.identity_map)
        self.phi_BA = self.regis_net(image_B, image_A)
        self.phi_BA_vectorfield = self.phi_BA(self.identity_map)

        # Add cycle edge
        self.phi_BcA = self.regis_net(cano_B, image_A)
        self.phi_BAc = self.regis_net(image_B, cano_A)
        if self.num_cano == "-1":
            self.phi_AcBc = self.regis_net(cano_A, cano_B)
        
        # For Mono-modal similarity
        self.phi_AAc_vectorfield = self.phi_BAc(self.phi_AB(self.identity_map))
        if self.num_cano == "-1":
            self.phi_AcA_vectorfield = self.phi_BcA(self.phi_AcBc(self.identity_map))
        else:
            self.phi_AcA_vectorfield = self.phi_BcA(self.identity_map)
        
        if self.num_cano == "-1":
            self.phi_BBc_vectorfield = self.phi_AcBc(self.phi_BAc(self.identity_map))
        else:
            self.phi_BBc_vectorfield = self.phi_BAc(self.identity_map)
        self.phi_BcB_vectorfield = self.phi_AB(self.phi_BcA(self.identity_map))
        
        # warp images
        inbounds_tag = None
        if getattr(self.similarity, "isInterpolated", False):
            # tag images during warping so that the similarity measure
            # can use information about whether a sample is interpolated
            # or extrapolated
            inbounds_tag = torch.zeros([image_A.shape[0]] + [1] + list(image_A.shape[2:]), device=image_A.device)
            if len(self.input_shape) - 2 == 3:
                inbounds_tag[:, :, 1:-1, 1:-1, 1:-1] = 1.0
            elif len(self.input_shape) - 2 == 2:
                inbounds_tag[:, :, 1:-1, 1:-1] = 1.0
            else:
                inbounds_tag[:, :, 1:-1] = 1.0

        self.warped_image_A = compute_warped_image_multiNC(
            torch.cat([image_A, inbounds_tag], axis=1) if inbounds_tag is not None else image_A,
            self.phi_AB_vectorfield,
            self.spacing,
            1,
            zero_boundary=True
        )
        with torch.no_grad():
            self.warped_label_A = compute_warped_image_multiNC(
                torch.cat([label_A, inbounds_tag], axis=1) if inbounds_tag is not None else label_A,
                self.phi_AB_vectorfield,
                self.spacing,
                0,
                zero_boundary=True
            )
            if self.log_mono:
                self.warped_image_B = compute_warped_image_multiNC(
                    torch.cat([image_B, inbounds_tag], axis=1) if inbounds_tag is not None else image_B,
                    self.phi_BA_vectorfield,
                    self.spacing,
                    1,
                    zero_boundary=True
                )
        self.warped_image_A_to_Ac = compute_warped_image_multiNC(
            torch.cat([image_A, inbounds_tag], axis=1) if inbounds_tag is not None else image_A,
            self.phi_AAc_vectorfield,
            self.spacing,
            1,
            zero_boundary=True
        )
        self.warped_image_Ac_to_A = compute_warped_image_multiNC(
            torch.cat([cano_A, inbounds_tag], axis=1) if inbounds_tag is not None else cano_A,
            self.phi_AcA_vectorfield,
            self.spacing,
            1,
            zero_boundary=True
        )
        self.warped_image_B_to_Bc = compute_warped_image_multiNC(
            torch.cat([image_B, inbounds_tag], axis=1) if inbounds_tag is not None else image_B,
            self.phi_BBc_vectorfield,
            self.spacing,
            1,
            zero_boundary=True
        )
        self.warped_image_Bc_to_B = compute_warped_image_multiNC(
            torch.cat([cano_B, inbounds_tag], axis=1) if inbounds_tag is not None else cano_B,
            self.phi_BcB_vectorfield,
            self.spacing,
            1,
            zero_boundary=True
        )

        # for logging
        if self.log_mono:
            with torch.no_grad():
                mono_similarity_loss = self.similarity(self.warped_image_A, image_B) + self.similarity(self.warped_image_B, image_A)

        # similarity loss        
        similarity_loss = self.similarity(self.warped_image_A_to_Ac, cano_A) + self.similarity(self.warped_image_Ac_to_A, image_A) \
                        + self.similarity(self.warped_image_B_to_Bc, cano_B) + self.similarity(self.warped_image_Bc_to_B, image_B)

        # DICE score
        with torch.no_grad():
            total_dice = dice_score(self.warped_label_A[:,0], label_B[:,0], dice_logging=dice_logging)
            # log_dice_scores(class_wise_dice, unique_labels)
        
        # inverse consistency loss        
        direction_losses_inv = []

        Iepsilon_inv = self.create_Iepsilon()

        approximate_Iepsilon_inv =  self.phi_AB(self.phi_BA(Iepsilon_inv))
        inverse_consistency_error = Iepsilon_inv - approximate_Iepsilon_inv

        delta = 0.001
        direction_vectors_inv = self.get_direction_vectors(delta=delta)

        for d_inv in direction_vectors_inv:
            approximate_Iepsilon_d_inv = self.phi_AB(self.phi_BA(Iepsilon_inv + d_inv))
            inverse_consistency_error_d = Iepsilon_inv + d_inv - approximate_Iepsilon_d_inv
            grad_d_icon_error = (inverse_consistency_error - inverse_consistency_error_d) / delta
            direction_losses_inv.append(torch.mean(grad_d_icon_error**2))
            
        inverse_consistency_loss = sum(direction_losses_inv)
        
        
        # Canonical cycle consistency loss
        direction_losses_can = []

        Iepsilon_can = self.create_Iepsilon()

        if self.num_cano == "-1":
            approximate_Iepsilon_can = self.phi_BcA(self.phi_AcBc(self.phi_BAc(self.phi_AB(Iepsilon_can))))
        else:
            approximate_Iepsilon_can = self.phi_BcA(self.phi_BAc(self.phi_AB(Iepsilon_can)))
        can_cycle_consistency_error = Iepsilon_can - approximate_Iepsilon_can

        delta = 0.001
        direction_vectors_can = self.get_direction_vectors(delta=delta)

        for d_can in direction_vectors_can:
            if self.num_cano == "-1":
                approximate_Iepsilon_d_can = self.phi_BcA(self.phi_AcBc(self.phi_BAc(self.phi_AB(Iepsilon_can + d_can))))
            else:
                approximate_Iepsilon_d_can = self.phi_BcA(self.phi_BAc(self.phi_AB(Iepsilon_can + d_can)))
            can_cycle_consistency_error_d = Iepsilon_can + d_can - approximate_Iepsilon_d_can
            grad_can_cycle_consistency_error = (can_cycle_consistency_error - can_cycle_consistency_error_d) / delta
            direction_losses_can.append(torch.mean(grad_can_cycle_consistency_error**2))
            
        can_cycle_consistency_loss = sum(direction_losses_can)
        
        
        # total loss
        all_loss = self.lambda_inv * inverse_consistency_loss + self.lambda_can * can_cycle_consistency_loss + similarity_loss

        with torch.no_grad():
            transform_magnitude = torch.mean((self.identity_map - self.phi_AB_vectorfield) ** 2)
        
        if self.log_mono:
            return ICONLoss_can_mono(
                all_loss,
                similarity_loss,
                mono_similarity_loss,
                inverse_consistency_loss,
                can_cycle_consistency_loss,
                transform_magnitude,
                flips(self.phi_BA_vectorfield, in_percentage=True),
                total_dice
            )
        return ICONLoss_can(
            all_loss,
            similarity_loss,
            inverse_consistency_loss,
            can_cycle_consistency_loss,
            transform_magnitude,
            flips(self.phi_BA_vectorfield, in_percentage=True),
            total_dice
        )

    def clean(self):
        if self.log_mono:
            del self.warped_image_B
        del self.phi_AB, self.phi_BA, self.phi_AB_vectorfield, self.phi_BA_vectorfield, self.warped_image_A, self.warped_label_A
        del self.warped_image_A_to_Ac, self.warped_image_Ac_to_A, self.warped_image_B_to_Bc, self.warped_image_Bc_to_B
        del self.phi_BcA, self.phi_BAc, self.phi_AAc_vectorfield, self.phi_AcA_vectorfield, self.phi_BBc_vectorfield, self.phi_BcB_vectorfield            
            
class GradientICONSparse(network_wrappers.RegistrationModule):
    def __init__(self, args, network, similarity, use_label=False):

        super().__init__()

        self.regis_net = network
        self.lmbda = args.lambda_inv
        self.similarity = similarity
        self.use_label = use_label
        self.input_shape = args.input_shape

    def forward(self, image_A, image_B, cano_A=None, cano_B=None, label_A=None, label_B=None, dice_logging=False):
        assert self.identity_map.shape[2:] == image_A.shape[2:]
        assert self.identity_map.shape[2:] == image_B.shape[2:]
        if self.use_label:
            label_A = image_A if label_A is None else label_A
            label_B = image_B if label_B is None else label_B
            assert self.identity_map.shape[2:] == label_A.shape[2:]
            assert self.identity_map.shape[2:] == label_B.shape[2:]

        # Tag used elsewhere for optimization.
        # Must be set at beginning of forward b/c not preserved by .cuda() etc
        self.identity_map.isIdentity = True

        self.phi_AB = self.regis_net(image_A, image_B)
        self.phi_BA = self.regis_net(image_B, image_A)

        self.phi_AB_vectorfield = self.phi_AB(self.identity_map)
        self.phi_BA_vectorfield = self.phi_BA(self.identity_map)

        # tag images during warping so that the similarity measure
        # can use information about whether a sample is interpolated
        # or extrapolated

        if getattr(self.similarity, "isInterpolated", False):
            # tag images during warping so that the similarity measure
            # can use information about whether a sample is interpolated
            # or extrapolated
            inbounds_tag = torch.zeros([image_A.shape[0]] + [1] + list(image_A.shape[2:]), device=image_A.device)
            if len(self.input_shape) - 2 == 3:
                inbounds_tag[:, :, 1:-1, 1:-1, 1:-1] = 1.0
            elif len(self.input_shape) - 2 == 2:
                inbounds_tag[:, :, 1:-1, 1:-1] = 1.0
            else:
                inbounds_tag[:, :, 1:-1] = 1.0
        else:
            inbounds_tag = None

        self.warped_image_A = compute_warped_image_multiNC(
            torch.cat([image_A, inbounds_tag], axis=1) if inbounds_tag is not None else image_A,
            self.phi_AB_vectorfield,
            self.spacing,
            1,
            zero_boundary=True
        )
        self.warped_image_B = compute_warped_image_multiNC(
            torch.cat([image_B, inbounds_tag], axis=1) if inbounds_tag is not None else image_B,
            self.phi_BA_vectorfield,
            self.spacing,
            1,
            zero_boundary=True
        )
        
        with torch.no_grad():
            self.warped_label_A = compute_warped_image_multiNC(
                torch.cat([label_A, inbounds_tag], axis=1) if inbounds_tag is not None else label_A,
                self.phi_AB_vectorfield,
                self.spacing,
                0, # nearset
                zero_boundary=True
            )
                
            # self.warped_label_B = compute_warped_image_multiNC(
            #     torch.cat([label_B, inbounds_tag], axis=1) if inbounds_tag is not None else label_B,
            #     self.phi_BA_vectorfield,
            #     self.spacing,
            #     1,
            # )
            
            # similarity_loss = self.similarity(
            #     self.warped_label_A, label_B
            # ) + self.similarity(self.warped_label_B, label_A)

        similarity_loss = self.similarity(self.warped_image_A, image_B) + self.similarity(self.warped_image_B, image_A)

        with torch.no_grad():
            total_dice = dice_score(self.warped_label_A[:,0], label_B[:,0], dice_logging=dice_logging)
            # log_dice_scores(class_wise_dice, unique_labels)
        
        # epsilon for perturbed identity map
        if len(self.input_shape) - 2 == 3:
            Iepsilon = (
                self.identity_map
                + 2 * torch.randn(*self.identity_map.shape).to(config.device)
                / self.identity_map.shape[-1]
            )[:, :, ::2, ::2, ::2]
        elif len(self.input_shape) - 2 == 2:
            Iepsilon = (
                self.identity_map
                + 2 * torch.randn(*self.identity_map.shape).to(config.device)
                / self.identity_map.shape[-1]
            )[:, :, ::2, ::2]

        # compute squared Frobenius of Jacobian of icon error

        direction_losses = []

        approximate_Iepsilon = self.phi_AB(self.phi_BA(Iepsilon))

        inverse_consistency_error = Iepsilon - approximate_Iepsilon

        delta = 0.001

        if len(self.identity_map.shape) == 4:
            dx = torch.tensor([[[[delta]], [[0.0]]]]).to(config.device)
            dy = torch.tensor([[[[0.0]], [[delta]]]]).to(config.device)
            direction_vectors = (dx, dy)

        elif len(self.identity_map.shape) == 5:
            dx = torch.tensor([[[[[delta]]], [[[0.0]]], [[[0.0]]]]]).to(config.device)
            dy = torch.tensor([[[[[0.0]]], [[[delta]]], [[[0.0]]]]]).to(config.device)
            dz = torch.tensor([[[[0.0]]], [[[0.0]]], [[[delta]]]]).to(config.device)
            direction_vectors = (dx, dy, dz)
        elif len(self.identity_map.shape) == 3:
            dx = torch.tensor([[[delta]]]).to(config.device)
            direction_vectors = (dx,)

        for d in direction_vectors:
            approximate_Iepsilon_d = self.phi_AB(self.phi_BA(Iepsilon + d))
            inverse_consistency_error_d = Iepsilon + d - approximate_Iepsilon_d
            grad_d_icon_error = (
                inverse_consistency_error - inverse_consistency_error_d
            ) / delta
            direction_losses.append(torch.mean(grad_d_icon_error**2))

        inverse_consistency_loss = sum(direction_losses)

        all_loss = self.lmbda * inverse_consistency_loss + similarity_loss

        with torch.no_grad():
            transform_magnitude = torch.mean(
                (self.identity_map - self.phi_AB_vectorfield) ** 2
            )

        return ICONLoss(
            all_loss,
            similarity_loss,
            inverse_consistency_loss,
            transform_magnitude,
            flips(self.phi_BA_vectorfield, in_percentage=True),
            total_dice
        )

    def clean(self):
        del self.phi_AB, self.phi_BA, self.phi_AB_vectorfield, self.phi_BA_vectorfield, self.warped_image_A, self.warped_image_B
        del self.warped_label_A

class TransMorph_wrapper_M2M(network_wrappers.RegistrationModule):
    def __init__(self, args, network, similarity, use_label=False):

        super().__init__()

        self.regis_net = network
        self.lambda_inv = args.lambda_inv
        self.lambda_can = args.lambda_can
        self.similarity = similarity
        self.use_label = use_label
        self.input_shape = args.input_shape
        self.num_cano = args.num_cano
        self.diffusion = Grad3d(penalty='l2')

    def create_Iepsilon(self):
        noise = 2 * torch.randn(*self.identity_map.shape).to(config.device) / self.identity_map.shape[-1]
        if len(self.input_shape) - 2 == 3:
            return (self.identity_map + noise)[:, :, ::2, ::2, ::2]
        elif len(self.input_shape) - 2 == 2:
            return (self.identity_map + noise)[:, :, ::2, ::2]
        return self.identity_map + noise

    def get_direction_vectors(self, delta=0.001):
        if len(self.identity_map.shape) == 4:
            return (torch.tensor([[[[delta]], [[0.0]]]]).to(config.device),
                    torch.tensor([[[[0.0]], [[delta]]]]).to(config.device))
        elif len(self.identity_map.shape) == 5:
            return (torch.tensor([[[[[delta]]], [[[0.0]]], [[[0.0]]]]]).to(config.device),
                    torch.tensor([[[[[0.0]]], [[[delta]]], [[[0.0]]]]]).to(config.device),
                    torch.tensor([[[[0.0]]], [[[0.0]]], [[[delta]]]]).to(config.device))
        return (torch.tensor([[[delta]]]).to(config.device),)

    def forward(self, image_A, image_B, cano_A, cano_B, label_A=None, label_B=None, dice_logging=False):
        assert self.identity_map.shape[2:] == image_A.shape[2:]
        assert self.identity_map.shape[2:] == image_B.shape[2:]
        assert self.identity_map.shape[2:] == cano_A.shape[2:]
        assert self.identity_map.shape[2:] == cano_B.shape[2:]

        self.identity_map.isIdentity = True
        self.phi_AB = self.regis_net(image_A, image_B)
        self.phi_AB_vectorfield = self.phi_AB(self.identity_map)
        # self.phi_BA = self.regis_net(image_B, image_A)
        # self.phi_BA_vectorfield = self.phi_BA(self.identity_map)

        # Add cycle edge
        self.phi_BcA = self.regis_net(cano_B, image_A)
        self.phi_BAc = self.regis_net(image_B, cano_A)
        if self.num_cano == "-1":
            self.phi_AcBc = self.regis_net(cano_A, cano_B)
        
        # For Mono-modal similarity
        self.phi_AAc_vectorfield = self.phi_BAc(self.phi_AB(self.identity_map))
        if self.num_cano == "-1":
            self.phi_AcA_vectorfield = self.phi_BcA(self.phi_AcBc(self.identity_map))
        else:
            self.phi_AcA_vectorfield = self.phi_BcA(self.identity_map)
        
        if self.num_cano == "-1":
            self.phi_BBc_vectorfield = self.phi_AcBc(self.phi_BAc(self.identity_map))
        else:
            self.phi_BBc_vectorfield = self.phi_BAc(self.identity_map)
        self.phi_BcB_vectorfield = self.phi_AB(self.phi_BcA(self.identity_map))
        
        # warp images
        inbounds_tag = None
        if getattr(self.similarity, "isInterpolated", False):
            # tag images during warping so that the similarity measure
            # can use information about whether a sample is interpolated
            # or extrapolated
            inbounds_tag = torch.zeros([image_A.shape[0]] + [1] + list(image_A.shape[2:]), device=image_A.device)
            if len(self.input_shape) - 2 == 3:
                inbounds_tag[:, :, 1:-1, 1:-1, 1:-1] = 1.0
            elif len(self.input_shape) - 2 == 2:
                inbounds_tag[:, :, 1:-1, 1:-1] = 1.0
            else:
                inbounds_tag[:, :, 1:-1] = 1.0

        self.warped_image_A = compute_warped_image_multiNC(
            torch.cat([image_A, inbounds_tag], axis=1) if inbounds_tag is not None else image_A,
            self.phi_AB_vectorfield,
            self.spacing,
            1,
            zero_boundary=True
        )
        with torch.no_grad():
            self.warped_label_A = compute_warped_image_multiNC(
                torch.cat([label_A, inbounds_tag], axis=1) if inbounds_tag is not None else label_A,
                self.phi_AB_vectorfield,
                self.spacing,
                0,
                zero_boundary=True
            )
        self.warped_image_A_to_Ac = compute_warped_image_multiNC(
            torch.cat([image_A, inbounds_tag], axis=1) if inbounds_tag is not None else image_A,
            self.phi_AAc_vectorfield,
            self.spacing,
            1,
            zero_boundary=True
        )
        self.warped_image_Ac_to_A = compute_warped_image_multiNC(
            torch.cat([cano_A, inbounds_tag], axis=1) if inbounds_tag is not None else cano_A,
            self.phi_AcA_vectorfield,
            self.spacing,
            1,
            zero_boundary=True
        )
        self.warped_image_B_to_Bc = compute_warped_image_multiNC(
            torch.cat([image_B, inbounds_tag], axis=1) if inbounds_tag is not None else image_B,
            self.phi_BBc_vectorfield,
            self.spacing,
            1,
            zero_boundary=True
        )
        self.warped_image_Bc_to_B = compute_warped_image_multiNC(
            torch.cat([cano_B, inbounds_tag], axis=1) if inbounds_tag is not None else cano_B,
            self.phi_BcB_vectorfield,
            self.spacing,
            1,
            zero_boundary=True
        )


        # similarity loss        
        similarity_loss = self.similarity(self.warped_image_A_to_Ac, cano_A) + self.similarity(self.warped_image_Ac_to_A, image_A) \
                        + self.similarity(self.warped_image_B_to_Bc, cano_B) + self.similarity(self.warped_image_Bc_to_B, image_B)

        # DICE score
        with torch.no_grad():
            total_dice = dice_score(self.warped_label_A[:,0], label_B[:,0], dice_logging=dice_logging)
            # log_dice_scores(class_wise_dice, unique_labels)
        
        # inverse consistency loss        
        inverse_consistency_loss = self.diffusion(self.phi_AB_vectorfield)
        
        # Canonical cycle consistency loss        
        direction_losses_can = []

        Iepsilon_can = self.create_Iepsilon()

        if self.num_cano == "-1":
            approximate_Iepsilon_can = self.phi_BcA(self.phi_AcBc(self.phi_BAc(self.phi_AB(Iepsilon_can))))
        else:
            approximate_Iepsilon_can = self.phi_BcA(self.phi_BAc(self.phi_AB(Iepsilon_can)))
        can_cycle_consistency_error = Iepsilon_can - approximate_Iepsilon_can

        delta = 0.001
        direction_vectors_can = self.get_direction_vectors(delta=delta)

        for d_can in direction_vectors_can:
            if self.num_cano == "-1":
                approximate_Iepsilon_d_can = self.phi_BcA(self.phi_AcBc(self.phi_BAc(self.phi_AB(Iepsilon_can + d_can))))
            else:
                approximate_Iepsilon_d_can = self.phi_BcA(self.phi_BAc(self.phi_AB(Iepsilon_can + d_can)))
            can_cycle_consistency_error_d = Iepsilon_can + d_can - approximate_Iepsilon_d_can
            grad_can_cycle_consistency_error = (can_cycle_consistency_error - can_cycle_consistency_error_d) / delta
            direction_losses_can.append(torch.mean(grad_can_cycle_consistency_error**2))
            
        can_cycle_consistency_loss = sum(direction_losses_can)
        
        
        # total loss
        all_loss = self.lambda_inv * inverse_consistency_loss + self.lambda_can * can_cycle_consistency_loss + similarity_loss

        with torch.no_grad():
            transform_magnitude = torch.mean((self.identity_map - self.phi_AB_vectorfield) ** 2)
        
        return ICONLoss_can(
            all_loss,
            similarity_loss,
            inverse_consistency_loss,
            can_cycle_consistency_loss,
            transform_magnitude,
            flips(self.phi_AB_vectorfield, in_percentage=True),
            total_dice
        )

    def clean(self):
        del self.phi_AB, self.phi_AB_vectorfield, self.warped_image_A, self.warped_label_A
        del self.warped_image_A_to_Ac, self.warped_image_Ac_to_A, self.warped_image_B_to_Bc, self.warped_image_Bc_to_B
        del self.phi_BcA, self.phi_BAc, self.phi_AAc_vectorfield, self.phi_AcA_vectorfield, self.phi_BBc_vectorfield, self.phi_BcB_vectorfield            
            
class TransMorph_wrapper(network_wrappers.RegistrationModule):
    def __init__(self, args, network, similarity, use_label=False):

        super().__init__()

        self.regis_net = network
        self.lmbda = args.lambda_inv
        self.similarity = similarity
        self.use_label = use_label
        self.input_shape = args.input_shape
        self.diffusion = Grad3d(penalty='l2')

    def forward(self, image_A, image_B, cano_A=None, cano_B=None, label_A=None, label_B=None, dice_logging=False):
        assert self.identity_map.shape[2:] == image_A.shape[2:]
        assert self.identity_map.shape[2:] == image_B.shape[2:]
        if self.use_label:
            label_A = image_A if label_A is None else label_A
            label_B = image_B if label_B is None else label_B
            assert self.identity_map.shape[2:] == label_A.shape[2:]
            assert self.identity_map.shape[2:] == label_B.shape[2:]

        # Tag used elsewhere for optimization.
        # Must be set at beginning of forward b/c not preserved by .cuda() etc
        self.identity_map.isIdentity = True

        self.phi_AB = self.regis_net(image_A, image_B)
        # self.phi_BA = self.regis_net(image_B, image_A)

        self.phi_AB_vectorfield = self.phi_AB(self.identity_map)
        # self.phi_BA_vectorfield = self.phi_BA(self.identity_map)

        # tag images during warping so that the similarity measure
        # can use information about whether a sample is interpolated
        # or extrapolated

        if getattr(self.similarity, "isInterpolated", False):
            # tag images during warping so that the similarity measure
            # can use information about whether a sample is interpolated
            # or extrapolated
            inbounds_tag = torch.zeros([image_A.shape[0]] + [1] + list(image_A.shape[2:]), device=image_A.device)
            if len(self.input_shape) - 2 == 3:
                inbounds_tag[:, :, 1:-1, 1:-1, 1:-1] = 1.0
            elif len(self.input_shape) - 2 == 2:
                inbounds_tag[:, :, 1:-1, 1:-1] = 1.0
            else:
                inbounds_tag[:, :, 1:-1] = 1.0
        else:
            inbounds_tag = None

        self.warped_image_A = compute_warped_image_multiNC(
            torch.cat([image_A, inbounds_tag], axis=1) if inbounds_tag is not None else image_A,
            self.phi_AB_vectorfield,
            self.spacing,
            1,
            zero_boundary=True
        )
        # self.warped_image_B = compute_warped_image_multiNC(
        #     torch.cat([image_B, inbounds_tag], axis=1) if inbounds_tag is not None else image_B,
        #     self.phi_BA_vectorfield,
        #     self.spacing,
        #     1,
        #     zero_boundary=True
        # )
        
        with torch.no_grad():
            self.warped_label_A = compute_warped_image_multiNC(
                torch.cat([label_A, inbounds_tag], axis=1) if inbounds_tag is not None else label_A,
                self.phi_AB_vectorfield,
                self.spacing,
                0, # nearset
                zero_boundary=True
            )
                
            # self.warped_label_B = compute_warped_image_multiNC(
            #     torch.cat([label_B, inbounds_tag], axis=1) if inbounds_tag is not None else label_B,
            #     self.phi_BA_vectorfield,
            #     self.spacing,
            #     1,
            # )
            
            # similarity_loss = self.similarity(
            #     self.warped_label_A, label_B
            # ) + self.similarity(self.warped_label_B, label_A)

        similarity_loss = self.similarity(self.warped_image_A, image_B)

        with torch.no_grad():
            total_dice = dice_score(self.warped_label_A[:,0], label_B[:,0], dice_logging=dice_logging)
            # log_dice_scores(class_wise_dice, unique_labels)

        inverse_consistency_loss = self.diffusion(self.phi_AB_vectorfield)

        all_loss = self.lmbda * inverse_consistency_loss + similarity_loss

        with torch.no_grad():
            transform_magnitude = torch.mean(
                (self.identity_map - self.phi_AB_vectorfield) ** 2
            )

        return ICONLoss(
            all_loss,
            similarity_loss,
            inverse_consistency_loss,
            transform_magnitude,
            flips(self.phi_AB_vectorfield, in_percentage=True),
            total_dice
        )

    def clean(self):
        del self.phi_AB, self.phi_AB_vectorfield, self.warped_image_A
        del self.warped_label_A

class Grad3d(torch.nn.Module):
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        super(Grad3d, self).__init__()
        self.penalty = penalty
        self.loss_mult = loss_mult

    def forward(self, y_pred):
        dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
        dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
        dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        grad = d / 3.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad
    

def log_dice_scores(dice_scores, unique_labels):
    """
    Logs class-wise Dice coefficients in a formatted table.
    
    Args:
        dice_scores (Tensor): Dice coefficient for each class. Shape (C,)
        unique_labels (Tensor): Corresponding class labels.
    """
    dice_dict = {int(label.item()): round(dice.item(), 4) for label, dice in zip(unique_labels.cpu(), dice_scores.cpu())}
    sorted_dice = sorted(dice_dict.items())

    logging.info("\n" + "=" * 40)
    logging.info(f"{'Class ID':<10}{'Dice Score':<10}")
    logging.info("-" * 40)
    for class_id, dice in sorted_dice:
        logging.info(f"{class_id:<10}{dice:<10.4f}")
    logging.info("=" * 40)
    

def dice_score(pred_label: torch.Tensor, target_label: torch.Tensor, smooth: float = 1e-6, dice_logging=False) -> torch.Tensor:
    """
    Compute multi-class Dice coefficient for segmentation maps with contiguous class values.

    Args:
        pred_label (Tensor): Predicted segmentation map of shape (N, D, H, W) with class indices.
        target_label (Tensor): Ground truth segmentation map of shape (N, D, H, W) with class indices.
        smooth (float): Smoothing factor to avoid division by zero.

    Returns:
        Tensor: Dice score averaged over all foreground classes (excluding background class 0).
    """
    pred_label = pred_label.long()
    target_label = target_label.long()
    num_classes = (max(pred_label.max(), target_label.max()) + 1).item()
    # print(num_classes)
    pred_one_hot = torch.nn.functional.one_hot(pred_label, num_classes)[:, :, :, :, 1:].permute(0, 4, 1, 2, 3).float() # exclude class 0
    target_one_hot = torch.nn.functional.one_hot(target_label, num_classes)[:, :, :, :, 1:].permute(0, 4, 1, 2, 3).float()
    
    intersection = (pred_one_hot * target_one_hot).sum(dim=(2, 3, 4))
    union = pred_one_hot.sum(dim=(2, 3, 4)) + target_one_hot.sum(dim=(2, 3, 4))
    dice = (2.0 * intersection + smooth) / (union + smooth)

    if dice_logging:
        unique_labels = torch.arange(1, num_classes, device=dice.device)
        log_dice_scores(dice.mean(dim=0), unique_labels)

    return dice.mean()  # Average over all batches and all classes

def get_3DTransMorph_config(args):
    '''
    Trainable params: 15,201,579
    '''
    config = ml_collections.ConfigDict()
    config.if_transskip = True
    config.if_convskip = True
    config.patch_size = 4
    config.in_chans = 2
    config.embed_dim = 96
    config.depths = (2, 2, 4, 2)
    config.num_heads = (4, 4, 8, 8)
    config.window_size = (5, 6, 7)
    config.mlp_ratio = 4
    config.pat_merg_rf = 4
    config.qkv_bias = False
    config.drop_rate = 0
    config.drop_path_rate = 0.3
    config.ape = False
    config.spe = False
    config.rpe = True
    config.patch_norm = True
    config.use_checkpoint = False
    config.out_indices = (0, 1, 2, 3)
    config.reg_head_chan = 16
    config.img_size = tuple(args.input_shape[2:])
    return config

def make_network(args, include_last_step=False, loss_fn=icon.LNCC(sigma=5), use_label=False):
    dimension = len(args.input_shape) - 2
    if args.model == 'transmorph':
        from other_models.TransMorph import TransMorph
        config = get_3DTransMorph_config(args)
        inner_net = icon.FunctionFromVectorField(TransMorph(config))
    elif args.model == 'corrmlp':
        from other_models.CorrMLP import CorrMLP
        inner_net = icon.FunctionFromVectorField(CorrMLP(enc_channels=4, dec_channels=8))
        # inner_net = icon.FunctionFromVectorField(CorrMLP())
    elif args.model == 'gradicon':
        if args.small:
            inner_net = icon.FunctionFromVectorField(networks.tallUNet2_small(dimension=dimension))

            for _ in range(2):
                inner_net = icon.TwoStepRegistration(
                    icon.DownsampleRegistration(inner_net, dimension=dimension),
                    icon.FunctionFromVectorField(networks.tallUNet2_small(dimension=dimension))
                )
            if include_last_step:
                inner_net = icon.TwoStepRegistration(inner_net, icon.FunctionFromVectorField(networks.tallUNet2_small(dimension=dimension)))
        else:
            inner_net = icon.FunctionFromVectorField(networks.tallUNet2(dimension=dimension))

            for _ in range(2):
                inner_net = icon.TwoStepRegistration(
                    icon.DownsampleRegistration(inner_net, dimension=dimension),
                    icon.FunctionFromVectorField(networks.tallUNet2(dimension=dimension))
                )
            if include_last_step:
                inner_net = icon.TwoStepRegistration(inner_net, icon.FunctionFromVectorField(networks.tallUNet2(dimension=dimension)))


    if args.model == 'gradicon':
        if args.num_cano == '0':
            net = GradientICONSparse(args, inner_net, loss_fn, use_label=use_label)
        else:
            net = GradientICONSparse_M2M(args, inner_net, loss_fn, use_label=use_label)
    else:
        if args.num_cano == '0':
            net = TransMorph_wrapper(args, inner_net, loss_fn, use_label=use_label)
        else:
            net = TransMorph_wrapper_M2M(args, inner_net, loss_fn, use_label=use_label)
        
    net.assign_identity_map(args.input_shape)
    return net

def make_sim(similarity):
    if similarity == "lncc":
        return icon.LNCC(sigma=5)
    elif similarity == "lncc2":
        return icon. SquaredLNCC(sigma=5)
    elif similarity == "mind":
        return icon.MINDSSC(radius=2, dilation=2)
    else:
        raise ValueError(f"Similarity measure {similarity} not recognized. Choose from [lncc, lncc2, mind].")

def get_multigradicon(loss_fn=icon.LNCC(sigma=5)):
    net = make_network(input_shape, include_last_step=True, loss_fn=loss_fn)
    from os.path import exists
    weights_location = "network_weights/multigradicon1.0/Step_2_final.trch"
    if not exists(weights_location):
        print("Downloading pretrained multigradicon model")
        import urllib.request
        import os
        download_path = "https://github.com/uncbiag/uniGradICON/releases/download/multigradicon_weights/Step_2_final.trch"
        os.makedirs("network_weights/multigradicon1.0/", exist_ok=True)
        urllib.request.urlretrieve(download_path, weights_location)
    print(f"Loading weights from {weights_location}")
    trained_weights = torch.load(weights_location, map_location=torch.device("cpu"), weights_only=True)
    net.regis_net.load_state_dict(trained_weights)
    net.to(config.device)
    net.eval()
    return net

def get_unigradicon(loss_fn=icon.LNCC(sigma=5)):
    net = make_network(input_shape, include_last_step=True, loss_fn=loss_fn)
    from os.path import exists
    weights_location = "network_weights/unigradicon1.0/Step_2_final.trch"
    if not exists(weights_location):
        print("Downloading pretrained unigradicon model")
        import urllib.request
        import os
        download_path = "https://github.com/uncbiag/uniGradICON/releases/download/unigradicon_weights/Step_2_final.trch"
        os.makedirs("network_weights/unigradicon1.0/", exist_ok=True)
        urllib.request.urlretrieve(download_path, weights_location)
    trained_weights = torch.load(weights_location, map_location=torch.device("cpu"), weights_only=True)
    net.regis_net.load_state_dict(trained_weights)
    net.to(config.device)
    net.eval()
    return net

def get_model_from_model_zoo(model_name="unigradicon", loss_fn=icon.LNCC(sigma=5)):
    if model_name == "unigradicon":
        return get_unigradicon(loss_fn)
    elif model_name == "multigradicon":
        return get_multigradicon(loss_fn)
    else:
        raise ValueError(f"Model {model_name} not recognized. Choose from [unigradicon, multigradicon].")

def quantile(arr: torch.Tensor, q):
    arr = arr.flatten()
    l = len(arr)
    return torch.kthvalue(arr, int(q * l)).values

def apply_mask(image, segmentation):
    segmentation_cast_filter = itk.CastImageFilter[type(segmentation),
                                            itk.Image.F3].New()
    segmentation_cast_filter.SetInput(segmentation)
    segmentation_cast_filter.Update()
    segmentation = segmentation_cast_filter.GetOutput()
    mask_filter = itk.MultiplyImageFilter[itk.Image.F3, itk.Image.F3,
                                    itk.Image.F3].New()

    mask_filter.SetInput1(image)
    mask_filter.SetInput2(segmentation)
    mask_filter.Update()

    return mask_filter.GetOutput()

def preprocess(image, modality="ct", segmentation=None):
    if modality == "ct":
        min_ = -1000
        max_ = 1000
        image = itk.CastImageFilter[type(image), itk.Image[itk.F, 3]].New()(image)
        image = itk.clamp_image_filter(image, Bounds=(min_, max_))
    elif modality == "mri":
        image = itk.CastImageFilter[type(image), itk.Image[itk.F, 3]].New()(image)
        min_, _ = itk.image_intensity_min_max(image)
        max_ = quantile(torch.tensor(np.array(image)), .99).item()
        image = itk.clamp_image_filter(image, Bounds=(min_, max_))
    else:
        raise ValueError(f"{modality} not recognized. Use 'ct' or 'mri'.")

    image = itk.shift_scale_image_filter(image, shift=-min_, scale = 1/(max_-min_)) 

    if segmentation is not None:
        image = apply_mask(image, segmentation)
    return image

def main():
    import itk
    import argparse
    parser = argparse.ArgumentParser(description="Register two images using unigradicon.")
    parser.add_argument("--fixed", required=True, type=str,
                         help="The path of the fixed image.")
    parser.add_argument("--moving", required=True, type=str,
                         help="The path of the fixed image.")
    parser.add_argument("--fixed_modality", required=True,
                         type=str, help="The modality of the fixed image. Should be 'ct' or 'mri'.")
    parser.add_argument("--moving_modality", required=True,
                         type=str, help="The modality of the moving image. Should be 'ct' or 'mri'.")
    parser.add_argument("--fixed_segmentation", required=False,
                         type=str, help="The path of the segmentation map of the fixed image. \
                         This map will be applied to the fixed image before registration.")
    parser.add_argument("--moving_segmentation", required=False,
                         type=str, help="The path of the segmentation map of the moving image. \
                         This map will be applied to the moving image before registration.")
    parser.add_argument("--transform_out", required=True,
                         type=str, help="The path to save the transform.")
    parser.add_argument("--warped_moving_out", required=False,
                        default=None, type=str, help="The path to save the warped image.")
    parser.add_argument("--io_iterations", required=False,
                         default="50", help="The number of IO iterations. Default is 50. Set to 'None' to disable IO.")
    parser.add_argument("--io_sim", required=False,
                         default="lncc", help="The similarity measure used in IO. Default is LNCC. Choose from [lncc, lncc2, mind].")
    parser.add_argument("--model", required=False,
                         default="unigradicon", help="The model to load. Default is unigradicon. Choose from [unigradicon, multigradicon].")

    args = parser.parse_args()

    net = get_model_from_model_zoo(args.model, make_sim(args.io_sim))

    fixed = itk.imread(args.fixed)
    moving = itk.imread(args.moving)
    
    if args.fixed_segmentation is not None:
        fixed_segmentation = itk.imread(args.fixed_segmentation)
    else:
        fixed_segmentation = None
    
    if args.moving_segmentation is not None:
        moving_segmentation = itk.imread(args.moving_segmentation)
    else:
        moving_segmentation = None

    if args.io_iterations == "None":
        io_iterations = None
    else:
        io_iterations = int(args.io_iterations)

    phi_AB, phi_BA = icon_registration.itk_wrapper.register_pair(
        net,
        preprocess(moving, args.moving_modality, moving_segmentation), 
        preprocess(fixed, args.fixed_modality, fixed_segmentation), 
        finetune_steps=io_iterations)

    itk.transformwrite([phi_AB], args.transform_out)

    if args.warped_moving_out:
        moving, maybe_cast_back = maybe_cast(moving)
        interpolator = itk.LinearInterpolateImageFunction.New(moving)
        warped_moving_image = itk.resample_image_filter(
                moving,
                transform=phi_AB,
                interpolator=interpolator,
                use_reference_image=True,
                reference_image=fixed
                )
        warped_moving_image = maybe_cast_back(warped_moving_image)
        itk.imwrite(warped_moving_image, args.warped_moving_out)

def warp_command():
    import itk
    import argparse
    parser = argparse.ArgumentParser(description="Warp an image with given transformation.")
    parser.add_argument("--fixed", required=True, type=str,
                         help="The path of the fixed image.")
    parser.add_argument("--moving", required=True, type=str,
                         help="The path of the moving image.")
    parser.add_argument("--transform")
    parser.add_argument("--warped_moving_out", required=True)
    parser.add_argument('--nearest_neighbor', action='store_true')
    parser.add_argument('--linear', action='store_true')

    args = parser.parse_args()

    fixed = itk.imread(args.fixed)
    moving = itk.imread(args.moving)
    if not args.transform:
        phi_AB = itk.IdentityTransform[itk.D, 3].New()
    else:
        phi_AB = itk.transformread(args.transform)[0]

    moving, maybe_cast_back = maybe_cast(moving)

    if args.linear:
        interpolator = itk.LinearInterpolateImageFunction.New(moving)
    elif args.nearest_neighbor:
        interpolator = itk.NearestNeighborInterpolateImageFunction.New(moving)
    else:
        raise Exception("Specify --nearest_neighbor or --linear")
    warped_moving_image = itk.resample_image_filter(
            moving,
            transform=phi_AB,
            interpolator=interpolator,
            use_reference_image=True,
            reference_image=fixed
            )
    
    warped_moving_image = maybe_cast_back(warped_moving_image)

    itk.imwrite(warped_moving_image, args.warped_moving_out)

def maybe_cast(img: itk.Image):
    """
    If an itk image is of a type that can't be used with InterpolateImageFunctions, cast it 
    and be able to cast it back
    """
    maybe_cast_back = lambda x: x

    if str((type(img), itk.D)) not in itk.NearestNeighborInterpolateImageFunction.GetTypesAsList():

        if type(img) in (itk.Image[itk.ULL, 3], itk.Image[itk.UL, 3]):
            raise Exception("Label maps of type unsigned long may have values that cannot be represented in a double")
 
        maybe_cast_back = itk.CastImageFilter[itk.Image[itk.D, 3], type(img)].New()

        img = itk.CastImageFilter[type(img), itk.Image[itk.D, 3]].New()(img)

    return img, maybe_cast_back

def compute_jacobian_map_command():
    import itk
    import argparse
    parser = argparse.ArgumentParser(description="Compute the Jacobian map of a given transform.")
    parser.add_argument("--transform", required=True, type=str,
                            help="The path to the transform file.")
    parser.add_argument("--fixed", required=True, type=str,
                            help="The path to the fixed image that has been used in the registration.")
    parser.add_argument("--jacob", required=True, default=None, help="The path to the output Jacobian map, \
                        e.g. /path/to/output_jacobian.nii.gz")
    parser.add_argument("--log_jacob", required=False, default=None, help="The path to the output log Jacobian map. \
                            If not specified, the log Jacobian map will not be saved.")
    args = parser.parse_args()

    transform_file = args.transform
    fixed_img_file = args.fixed
    transform = itk.transformread(transform_file)[0]
    jacob = itk.displacement_field_jacobian_determinant_filter(
        itk.transform_to_displacement_field_filter(
            transform,
            reference_image=itk.imread(fixed_img_file),
            use_reference_image=True
        )
    )
    itk.imwrite(jacob, args.jacob)

    if args.log_jacob is not None:
        log_jacob = itk.LogImageFilter.New(jacob)
        log_jacob.Update()
        log_jacob = log_jacob.GetOutput()
        itk.imwrite(jacob, args.log_jacob)
