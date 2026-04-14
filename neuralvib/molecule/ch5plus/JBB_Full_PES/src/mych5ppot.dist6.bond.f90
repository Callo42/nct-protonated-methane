! contains
!  function ch5ppot_func 
!  subroutine polar2cart
!  subroutine distance
!******************************************************************************
!CH5+ Potential Energy Surface, version 7a, 2005-10-18
!   by Zhong Jin, Bastiaan J. Braams, Joel M. Bowman
! with modifications by Xiaogang Wang
!
! input r(12) is polyspherical coordinates --- bond vector assumed
! distance in bohr, angles in radian, energy in cm-1
!
! vceil = 20 K is forced on any rejected points 
! Jul22,2011 Xiaogang Wang
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! only use H-H distance to reject
!
!  write(21,'(a)') 'constrain version : dist6'
!  write(21,'(4(a/))')   ' dist_min = 1.12d0', &
!                        'dist2_min = 1.70d0', &
!                        'dist3_min = 1.90d0', &
!                        'dist4_min = 3.00d0'  
! The constraint was discussed in 
!  X.-G. Wang and T. Carrington Jr., J. Chem. Phys. 129 (2008) 234102 
!
  function ch5ppot_func_cart(cartr)
  implicit double precision (a-h,o-z)
  parameter (nvec=5, natom=6)
  parameter (ndist=15)
  parameter ( dist_min = 1.12d0)
  parameter (dist2_min = 1.70d0)
  parameter (dist3_min = 1.90d0) ! not very useful
  parameter (dist4_min = 3.00d0)
! equilibrium  r = 1.4 bohr, pot = -1.174 475
!              r = 1.2             -1.164 935   2090 cm-1
!              r = 1.0             -1.124 540  11000 cm-1
!              r = 0.8             -1.020 057  33900 cm-1
!
! at r = 2.14 bohr, theta = 30, r(H-H) = 1.11 
! at r = 2.14 bohr, theta = 40, r(H-H) = 1.46
! at r = 2.14 bohr, theta = 50, r(H-H) = 1.81

! at r = 2.14 bohr, theta = 60, r(H-H) = 2.14
! at r = 2.14 bohr, theta = 70, r(H-H) = 2.45

  real*8 :: cartr(3,nvec), cartb(3,nvec)
  real*8 :: cart2(3,natom)
!  real*8 :: xm(natom)
!  real*8 :: cth(nvec, nvec)
  real*8 :: rhh(nvec, nvec)
  real*8 :: dist(ndist)


  data zero/0d0/


  real*8, parameter :: wellde = -40.6527648702729d0  ! for pes7a
  real*8, parameter :: vceil = 0.09112669999


!  d2r = dacos(-1d0)/180d0
!  r2d = 180d0/dacos(-1d0)
  

! get Cartersians of Radau

  cartb = cartr


!  only use H-H distance to reject exotic points
  do i = 1, nvec-1
  do j = i+1, nvec
    x = cartb(1,i) - cartb(1,j) 
    y = cartb(2,i) - cartb(2,j) 
    z = cartb(3,i) - cartb(3,j) 
    fx = sqrt(x*x + y*y + z*z)
!    write(*,'(2i5,f15.5)')i-1,j-1,fx !*a0
    
    if ( fx < dist_min ) then
      ch5ppot_func_cart = vceil
      return
    end if
    rhh(i,j) = fx
    rhh(j,i) = fx
  end do
  end do

  do i = 1, nvec
    rhh(i,i) = 0d0
  end do


  do i = 1, nvec
    call sort(nvec, rhh(i,:))
!    write(*,'(5f10.2)') rhh(i,1:nvec)
    if ( rhh(i,3) < dist2_min) then 
      ch5ppot_func_cart = vceil
      return
    end if
    if ( rhh(i,4) < dist3_min) then 
      ch5ppot_func_cart = vceil
      return
    end if
    if ( rhh(i,5) < dist4_min) then 
      ch5ppot_func_cart = vceil
      return
    end if
 end do
 
! just ignores Bowman's potential and return zero.
! ch5ppot_func_cart = 0d0
! return

! the first atom is Carbon, go from vector Cartesians to atoms Cartesians
  do i = 1, 3
    cart2(i,1) = zero
  end do
  
  do j = 1, nvec
    do i = 1, 3
      cart2(i,j+1) = cartb(i,j)
    end do
  end do

!! return if any of the 15 distances are too small
!  call distance(natom, ndist, cart2, dist)
!  kclose = 0
!  do i = 1, ndist
!    if (dist(i) < dist_min ) then
!      kclose = kclose + 1
!!      rsmall = dist(i)
!    end if
!  end do
!  if (kclose > 0 ) then
!    ch5ppot_func_cart = vceil
!    return
!  end if


  call getpot(cart2, ch5ppot_func_cart)
!  call getfit(cart2, ch5ppot_func_cart) ! skip CH3+H2 test
  ch5ppot_func_cart = (ch5ppot_func_cart - wellde)

  end function ch5ppot_func_cart

