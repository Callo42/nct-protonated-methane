!***********************************************************************
!
!.... Subroutine to calculate R, which is the distance between C and the
!.... center of H2 in CH5+
!
!.....History:
!.... Date     Modified by  Comment
!.... ----     -----------  -------
!.... 02/20/05   Zhong      created
!.... 03/01/05   Zhong      refine it

!.... dist ....  R
!.... h2dist ... bond length of H2
!.... theta ...  the angle between H2 and axis of C-center of H2
! 
************************************************************************
      SUBROUTINE calcr(x,xch3,xh2a,dist,h2dist,theta)

      implicit none
      integer :: i, j, npp1, npp2, jr
      integer, parameter :: natoms = 6

      double precision, dimension(3,natoms) :: x
      double precision, dimension(natoms-1) :: rch
      double precision, dimension(3) :: xh2m, xh1, xh2
      double precision, dimension(2) :: rchmax
      double precision :: dist, h2dist, h2dist5, theta
      double precision, dimension(3,4) :: xch3t
      double precision, dimension(3,2) :: xh2at
      double precision, dimension(4,3) :: xch3
      double precision, dimension(2,3) :: xh2a
       
      do j = 1, natoms - 1
         rch(j) = sqrt((x(1,j+1)-x(1,1))**2+(x(2,j+1)-x(2,1))**2
     $            +(x(3,j+1)-x(3,1))**2)
      end do
      rchmax(1) = 0.d0
      rchmax(2) = 0.d0
      npp1 = 0
      npp2 = 0

      do j = 1, natoms - 1
         if (rchmax(1)<rch(j)) then
            rchmax(1) = rch(j)
            npp1 = j
         end if
      end do

      do j = 1, natoms - 1
         if (npp1==j) then
         else
            if (rchmax(2)<rch(j)) then
               rchmax(2) = rch(j)
               npp2 = j
            end if
         end if
      end do

!.... Find the Cartisian coordinate of the center of H2
      do i = 1, 3
         xh2m(i) = 0.5d0*(x(i,npp1+1)+x(i,npp2+1))
      end do

!.... Obtain the R -- dist

      call calcdist(x,xh2m,dist)

!.... The Cartisian coordinate of H2
      do i = 1, 3
         xh1(i) = x(i,npp1+1)
         xh2(i) = x(i,npp2+1)
      end do

      do i = 1, 3
         xh2at(i,1) = x(i,npp1+1)
         xh2at(i,2) = x(i,npp2+1)
      end do

      jr = 0
      do j = 1, natoms
         if (j==npp1+1.or.j==npp2+1) then
         else
            jr = jr + 1
            do i = 1, 3
               xch3t(i,jr) = x(i,j)
            end do
         end if
      end do

         
      do i = 1, 3
         do j = 1, 4
            xch3(j,i) = xch3t(i,j)
         end do
      end do

      do i = 1, 3
         do j = 1, 2
            xh2a(j,i) = xh2at(i,j)
         end do
      end do

!.... Calculate the bond distance of H2
      call calcdist(xh1,xh2,h2dist)

      h2dist5 = h2dist/2.d0
      call calcangle(rch(npp2),dist,h2dist5,theta)
      END SUBROUTINE calcr
