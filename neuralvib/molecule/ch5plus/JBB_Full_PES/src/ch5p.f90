!******************************************************************************
!  call the potential routine with polyspherical coordinates of bond vectors
! Jul22,2011 Xiaogang Wang
  program main
  implicit real*8 (a-h,o-z)
  parameter (np = 12)

  real*8 :: r(np)


  data hartree2wn/2.194746313710d5/ ! (17)

  a0 = 0.5291772083d0 ! (19) \AA
  d2r = dacos(-1d0)/180d0
  r2d = 180d0/dacos(-1d0)


  write(*,'(/a)')'******Call the potential using polar coordinates******'

! Optimized geometry for bond vectors
!     2.09459     2.05676     2.26113     2.26260     2.05676
!     109.248     119.433      97.451     116.374
!     126.722     229.665     274.835
!               ener=       0.0

  r(1:5)=(/   2.09459d0,     2.05676d0,     2.26113d0,     2.26260d0,     2.05676d0/)
  r( 6: 9)=(/109.248d0,     119.433d0,      97.451d0,     116.374d0/)*d2r
  r(10:12)=(/126.722d0,     229.665d0,     274.835d0/)*d2r
  ener = ch5ppot_func(r)
  write(*,901)'Bond equilibrium'
  write(*,911)r(1:5),r(6:12)*r2d
  write(*,'(a20,f12.3/)')'ener=',ener 

! C2v minimum for bond vectors
!     2.15785     2.05472     2.05472     2.15785     2.19719
!      61.566     118.973     118.973      61.566
!      90.000     270.000     180.000
!               ener=     340.759
  r(1:5)=(/   2.15785d0,     2.05472d0,     2.05472d0,     2.15785d0,     2.19719d0/)
  r( 6: 9)=(/61.566d0,     118.973d0,     118.973d0,      61.566d0/)*d2r
  r(10:12)=(/90.000d0,     270.000d0,     180.000d0/)*d2r
  ener = ch5ppot_func(r)
  write(*,901)'C2v local minimum'
  write(*,911)r(1:5),r(6:12)*r2d
  write(*,'(a20,f12.3/)')'ener=',ener !*hartree2wn


! DIY try
  r(1:5)=(/   2.0,     2.0,     2.0,     2.0,     2.0/)
  r( 6: 9)=(/61.566d0,     118.973d0,     118.973d0,      61.566d0/)*d2r
  r(10:12)=(/90.000d0,     270.000d0,     180.000d0/)*d2r
  ener = ch5ppot_func(r)
  write(*,901)'DIY try'
  write(*,911)r(1:5),r(6:12)*r2d
  write(*,'(a20,f12.3/)')'ener=',ener !*hartree2wn





  901 format(a,f15.2,/)
  902 format(a,f20.13,f15.5,f10.3/)
  911 format(5f12.5,/,4f12.3/,3f12.3)
  end




