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
  function ch5ppot_func(r)
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

  real*8, intent(in) :: r(12)
  real*8 :: cartr(3,nvec), cartb(3,nvec)
  real*8 :: cart2(3,natom)
!  real*8 :: xm(natom)
!  real*8 :: cth(nvec, nvec)
  real*8 :: rhh(nvec, nvec)
  real*8 :: dist(ndist)


  data zero/0d0/


  real*8, parameter :: wellde = -40.6527648702729d0  ! for pes7a
  real*8, parameter :: hartree2wn = 2.194746313710d5 ! (17)
  real*8, parameter :: vceil = 2d4


!  d2r = dacos(-1d0)/180d0
!  r2d = 180d0/dacos(-1d0)
  

! get Cartersians of Radau
  call polar2cart(r, cartr)

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
      ch5ppot_func = vceil
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
      ch5ppot_func = vceil
      return
    end if
    if ( rhh(i,4) < dist3_min) then 
      ch5ppot_func = vceil
      return
    end if
    if ( rhh(i,5) < dist4_min) then 
      ch5ppot_func = vceil
      return
    end if
 end do
 
! just ignores Bowman's potential and return zero.
! ch5ppot_func = 0d0
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
!    ch5ppot_func = vceil
!    return
!  end if


  call getpot(cart2, ch5ppot_func)
!  call getfit(cart2, ch5ppot_func) ! skip CH3+H2 test
  ch5ppot_func = (ch5ppot_func - wellde)*hartree2wn

  end function ch5ppot_func


! *****************************************************************************
! convert polyspherical for 4 vectors with the same origin
! into the Cartesians of the end points of the 4 vectors
!     1  -  R1
!     2  -  R2
!     3  -  R3
!     4  -  R4
!     5  -  R5
!     6  -  th1
!     7  -  th2
!     8  -  th3
!     9  -  th4
!    10  -  ph2, azimuthal angle of 2, 4-C-1 is x-z plane
!    11  -  ph3, azimuthal angle of 3
!    12  -  ph4, azimuthal angle of 4
!
!     5
!     |
!     |
!     !
!     C
!   / | \ \
!  /  |  \   \
! /   |   \    \
!1    2    3    4
!     My convention is pi >= ph2 >= 0, 2pi >= ph3,ph4 >=0
  subroutine polar2cart(r,cart)
  implicit double precision (a-h,o-z)

  real*8, intent(in) :: r(12)
  real*8, intent(out):: cart(3,5)

! =====================================
! calculate Cartesians of the end atoms of the 4 vectors

  th1 = r(6)
  th2 = r(7)
  th3 = r(8)
  th4 = r(9)
  ph2 = r(10)
  ph3 = r(11)
  ph4 = r(12)

! vector 5 parallel to z axis
  cart(1,5)=0d0
  cart(2,5)=0d0
  cart(3,5)=r(5)

! vector 1 on x-z plane
  cart(1,1)=r(1)*sin(th1)
  cart(2,1)=0d0
  cart(3,1)=r(1)*cos(th1)

! vector 2 defined by (th2,ph2)
  cart(1,2)=r(2)*sin(th2)*cos(ph2)
  cart(2,2)=r(2)*sin(th2)*sin(ph2)
  cart(3,2)=r(2)*cos(th2)

! vector 3 defined by (th3,ph3)
  cart(1,3)=r(3)*sin(th3)*cos(ph3)
  cart(2,3)=r(3)*sin(th3)*sin(ph3)
  cart(3,3)=r(3)*cos(th3)

! vector 4 defined by (th4,ph4)
  cart(1,4)=r(4)*sin(th4)*cos(ph4)
  cart(2,4)=r(4)*sin(th4)*sin(ph4)
  cart(3,4)=r(4)*cos(th4)

  end subroutine polar2cart  




! *****************************************************************************
! 
! ndist = natom*(natom-1)/2
  subroutine distance(natom,ndist, cart,dist)
  implicit double precision (a-h,o-z)

  integer, intent(in) :: natom, ndist
  real*8, intent(in) :: cart(3,natom)
  real*8, intent(out) :: dist(ndist)
  real*8 :: vec(3,5), rlen(5), cth(5,5)
  data a0/0.5291772083d0/
!  ndist = natom*(natom-1)/2
!  allocate(dist(ndist))
  
! =====================================
!  write(*,*)'all distances before sorting'
  id = 0
  do i=1,natom-1
  do j=i+1,natom
    id=id+1
    x = cart(1,i) - cart(1,j) 
    y = cart(2,i) - cart(2,j) 
    z = cart(3,i) - cart(3,j) 
    dist(id) = sqrt(x*x + y*y + z*z)
!    write(*,'(2i5,f15.5)')i-1,j-1,dist(id) !*a0
        
  end do
  end do

!  call sort(ndist, dist)
!  write(*,*)'all distances'
!  do i=1,ndist
!    write(*,'(i5,f15.5)')i,dist(i)*a0
!  end do
!
! =====================================
! get CH angles
!  do i = 2, natom
!    vec(:,i-1) = cart(:,i) - cart(:,1)
!  end do
!  nvec = 5
!  do i = 1, nvec
!    rlen(i) = sqrt(dot_product(vec(:,i), vec(:,i)))
!  end do

!  cth = 0d0
!  r2d = 180d0/dacos(-1d0)
!  do i = 1, nvec-1
!  do j = i+1, nvec
!    fx = dot_product(vec(:,i), vec(:,j))/(rlen(i)*rlen(j))
!    fx = dacos(fx)*r2d
!    cth(i,j) = fx
!    cth(j,i) = fx
!  end do
!  end do
!  write(*,*)  'angles'
!  do i = 1, nvec
!    write(*,'(5f10.2)') cth(i,1:5)
!  end do
  
  
  
  end subroutine distance


!**********************************************************************
! sort into ascending order
  subroutine sort(n, x)
  implicit double precision (a-h,o-z)
!  real*8 :: x(n),xw(n)
  real*8, intent(inout) :: x(n)

  do i=1,n-1
    k=i
    xmin=x(i)
    do j=i+1,n
      if(x(j).le.xmin)then
        k=j
        xmin=x(j)
      endif
    end do

    if(k.ne.i)then
!     interchange data between i and k
      x(k)=x(i)
      x(i)=xmin
!     interchange accompanying data
!      do j=1,n
!        xmin=a(j,k)
!        a(j,k)=a(j,i)
!        a(j,i)=xmin
!      end do

!      xmin=xw(k)
!      xw(k)=xw(i)
!      xw(i)=xmin
    endif
  end do

  end subroutine sort
