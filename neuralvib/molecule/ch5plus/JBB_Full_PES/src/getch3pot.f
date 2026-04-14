      SUBROUTINE getch3pot(xyz1,v)
!...  program is to calculate the potential energy of CH3+
!...  Based on pes6n4
!...
      implicit none
      integer, parameter :: natoms = 6
      integer, parameter :: natoms1 = 4
      integer :: i, j

      double precision, dimension(natoms1,3) :: xyz1
      double precision, dimension(natoms,3) :: xyz
      double precision, dimension(3,natoms) :: xyzt, f
      double precision :: v, h2pot, h2potab, ch30
      double precision, parameter :: ang = 219474.6d0

      character(len=1), dimension(natoms) :: xname

      do i = 1, natoms1
         do j = 1, 3
            xyz(i,j) = xyz1(i,j)
         end do
      end do

      xyz(5,1) = 0.7035d0
      xyz(5,2) = 0.d0
      xyz(5,3) = 14.0750143d0
      xyz(6,1) = -0.7035d0
      xyz(6,2) = 0.d0
      xyz(6,3) = 14.0750143d0

      do i = 1, natoms
         do j = 1, 3
            xyzt(j,i) = xyz(i,j)
         end do
      end do

      call getfit(xyzt,v)

      h2pot = -1.1745554d0 + 0.0006728d0/2.d0
!      h2potab = -1.17263405d0
      ch30 = -39.4046592d0 + 0.0006415d0/2.d0
!      ch30 = -39.40570072d0

      v = v - h2pot - ch30

      if ((v*ang.le.5.d0).and.(v*ang.ge.-5.d0)) then
         v = 0.d0
      end if
      END SUBROUTINE getch3pot
