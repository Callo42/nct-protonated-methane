      SUBROUTINE geth2pot(xyz1,v)
!...  program is to calculate the potential energy of H2
!...  Based on pes6n4
!...
      implicit none
      integer, parameter :: natoms = 6
      integer, parameter :: natoms1 = 2
      integer :: i, j

      double precision, dimension(5:6,3) :: xyz1
      double precision, dimension(natoms,3) :: xyz
      double precision, dimension(3,natoms) :: xyzt, f
      double precision :: v, ch3pot, ch3potab, h20
      double precision, parameter :: ang = 219474.6d0

      character(len=1), dimension(natoms) :: xname

      do i = 5, 6
         do j = 1, 3
            xyz(i,j) = xyz1(i,j)
         end do
      end do

      xyz(1,1) = 0.d0
      xyz(1,2) = 0.d0
      xyz(1,3) = 0.d0
      xyz(2,1) = 0.d0
      xyz(2,2) = 2.0657297d0
      xyz(2,3) = 0.d0
      xyz(3,1) = 1.7889744d0
      xyz(3,2) = -1.0328649d0
      xyz(3,3) = 0.d0
      xyz(4,1) = -1.7889744d0
      xyz(4,2) = -1.0328649d0
      xyz(4,3) = 0.d0

      do i = 1, natoms
         do j = 1, 3
            xyzt(j,i) = xyz(i,j)
         end do
      end do

      call getfit(xyzt,v)

      ch3pot = -39.4046592d0 + 0.0006415d0/2.d0

!      ch3potab = -39.40570072d0
      h20 = -1.1745554d0 + 0.0006728d0/2.d0

      v = v - ch3pot - h20
      if ((v*ang.le.5.d0).and.(v*ang.ge.-5.d0)) then
         v = 0.d0
      end if
      END SUBROUTINE geth2pot
