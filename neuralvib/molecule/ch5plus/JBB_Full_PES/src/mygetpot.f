!***********************************************************************
! Changed by RuisiWang, July4, 2024
! to return epot for a python module call.
!***********************************************************************
!***********************************************************************
! changes by xgw, Sep19,2005, again Apr12,2006
!  replace x with xn (which is used before)
!  remove r
!***********************************************************************
!CH5+ Potential Energy Surface, version 7a, 2005-10-18
!   by Zhong Jin, Bastiaan J. Braams, Joel M. Bowman
!
!reference:
!   Zhong Jin, Bastiaan J. Braams, Joel M. Bowman
!   Journal of Physical Chemistry A  accepted(2005)
!
!notes:
!
!   x(6,3) is the Cartesian coordinates for six atoms in order of
!   C H H H H H. (in bohr)
!
!   r is defined as the distance between the center of H2 and Carbon 
!   in a.u.
!
!   Potential energy epot is returned in a.u.
!
!   CCSD(T)/aug-cc-pVTZ based
!
!   When compiling, "-r8" option is required for accurate potential
!   energy calculation
!   
!All rights reserved. Contact bowman@euch4e.chem.emory.edu for
!details or any latest updates.
! 
************************************************************************
!      SUBROUTINE getpot(x,epot)
      FUNCTION getpot_with_return(xn)
!
      implicit none
!
      integer, parameter :: natoms = 6
      integer :: i, j, k, l, m

      double precision :: sw, a, b, theta
      double precision :: vfit, getpot_with_return
!      double precision, dimension(natoms,3) :: x
      double precision, dimension(3,natoms) :: xn
      double precision, dimension(4,3) :: xch3
      double precision, dimension(2,3) :: xh2
      double precision, parameter :: de = -40.57833477d0
      double precision :: epotch3, epoth2, elr
      double precision :: r, rh2, rp
      double precision, parameter :: rmin = 11.d0, rmax = 15.d0

!      do i = 1, natoms
!         do j = 1, 3
!            xn(j,i) = x(i,j)
!         end do
!      end do

      call calcr(xn,xch3,xh2,r,rh2,theta)

!      write(*,'(a2,2x,f9.2)') 'r=', r
      a = 0.d0
      b = 0.d0

      if (r<=rmax) then
         call getfit(xn,vfit)
      end if

      rp = (r - rmin)/(rmax - rmin)
!      write(*,*) 'rp=', rp
      if (r<rmin) then
         getpot_with_return = vfit
      else
         call getlr(r,rh2,theta,elr)
         call getch3pot(xch3,epotch3)
         call geth2pot(xh2,epoth2)

!         write(*,'a8,f9.3')'epotch3=', epotch3
!         write(*,'a7,1x,f9.3')'epoth2=', epoth2
!         write(*,'a10,e20.10')'longrange=',elr
         if (r>rmax) then
            getpot_with_return = elr + epoth2 + epotch3 + de
         else
            getpot_with_return = (1.d0-sw(a,b,rp))*vfit + sw(a,b,rp)*
     *             (elr + epoth2 + epotch3 + de)
         end if
      end if

      END FUNCTION getpot_with_return
